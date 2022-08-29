import datetime as dt
import math
from copy import deepcopy
from os.path import join

# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from dlib.parallel import MyDDP as DDP
from dlib.process.parseit import parse_input

from dlib.process.instantiators import get_model
from dlib.process.instantiators import get_optimizer
from dlib.utils.tools import log_device
from dlib.utils.tools import bye

from dlib.configure import constants
from dlib.learning.train_wsol import Trainer
from dlib.process.instantiators import get_loss
from dlib.process.instantiators import get_pretrainde_classifier
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc
from dlib.utils.utils_checkpoints import find_last_checkpoint
from dlib.utils.utils_checkpoints import load_checkpoint_net
from dlib.utils.utils_checkpoints import load_checkpoint_optimizer
from dlib.utils.utils_checkpoints import load_checkpoint_lr_scheduler
from dlib.utils.utils_checkpoints import load_loss_t

import dlib.dllogger as DLLogger


def main():
    args, args_dict = parse_input(eval=False)
    log_device(args)

    model = get_model(args)
    init_iter, checkpoint = find_last_checkpoint(
        join(args.outd_backup, args.save_dir_models), key=constants.CHP_CP)

    current_step = init_iter

    model.cuda(args.c_cudaid)
    load_checkpoint_net(network=model, s_dict=checkpoint[constants.CHP_M])

    if args.distributed:
        dist.barrier()

    model = DDP(model, device_ids=[args.c_cudaid])

    best_state_dict = deepcopy(model.state_dict())

    optimizer, lr_scheduler = get_optimizer(args, model)
    load_checkpoint_optimizer(optimizer=optimizer,
                              s_dict=checkpoint[constants.CHP_O])
    load_checkpoint_lr_scheduler(lr_scheduler=lr_scheduler,
                                 s_dict=checkpoint[constants.CHP_LR])
    loss = get_loss(args)
    load_loss_t(loss, s_t=checkpoint[constants.CHP_T])

    inter_classifier = None
    if args.task in [constants.F_CL, constants.C_BOX, constants.TCAM]:
        chpts = {
            constants.F_CL: constants.BEST_LOC,
            constants.C_BOX: args.cb_pretrained_cl_ch_pt,
            constants.TCAM: args.tcam_pretrained_seeder_ch_pt
        }
        inter_classifier = get_pretrainde_classifier(
            args, pretrained_ch_pt=chpts[args.task])
        inter_classifier.cuda(args.c_cudaid)

    trainer: Trainer = Trainer(
        args=args, model=model, optimizer=optimizer,
        lr_scheduler=lr_scheduler, loss=loss,
        classifier=inter_classifier, current_step=current_step)

    DLLogger.log(fmsg("Start init. epoch ..."))

    tr_loader = trainer.loaders[constants.TRAINSET]
    train_size = int(math.ceil(
        len(tr_loader.dataset) / (args.batch_size * args.num_gpus)))
    current_epoch = math.floor(current_step / float(train_size))

    trainer.evaluate(epoch=current_epoch, split=constants.VALIDSET)

    if args.is_master:
        trainer.model_selection(epoch=current_epoch, split=constants.VALIDSET)
        trainer.print_performances()
        trainer.report(epoch=0, split=constants.VALIDSET)

    DLLogger.log(fmsg("Epoch init. epoch done."))

    for epoch in range(current_epoch, trainer.args.max_epochs, 1):
        if args.distributed:
            dist.barrier()

        zepoch = epoch + 1
        DLLogger.log(fmsg(("Start epoch {} ...".format(zepoch))))

        train_performance = trainer.train(
            split=constants.TRAINSET, epoch=zepoch)

        trainer.evaluate(zepoch, split=constants.VALIDSET)

        if args.is_master:
            trainer.model_selection(epoch=zepoch, split=constants.VALIDSET)

            trainer.report_train(train_performance, zepoch,
                                 split=constants.TRAINSET)
            trainer.print_performances()
            trainer.report(zepoch, split=constants.VALIDSET)
            DLLogger.log(fmsg(("Epoch {} done.".format(zepoch))))

        trainer.adjust_learning_rate()

    if args.distributed:
        dist.barrier()

    trainer.save_best_epoch(split=constants.VALIDSET)
    trainer.capture_perf_meters()

    DLLogger.log(fmsg("Final epoch evaluation on test set ..."))

    chpts = [constants.BEST_LOC, constants.BEST_CL]
    # todo: keep only best_loc eval for tcam.

    if args.dataset == constants.ILSVRC:
        chpts = [constants.BEST_LOC]

    for eval_checkpoint_type in chpts:
        t0 = dt.datetime.now()

        DLLogger.log(fmsg('EVAL TEST SET. CHECKPOINT: {}'.format(
            eval_checkpoint_type)))

        if eval_checkpoint_type == constants.BEST_LOC:
            epoch = trainer.args.best_epoch_loc
        elif eval_checkpoint_type == constants.BEST_CL:
            epoch = trainer.args.best_epoch_cl
        else:
            raise NotImplementedError

        trainer.load_checkpoint(checkpoint_type=eval_checkpoint_type)

        trainer.evaluate(epoch,
                         split=constants.TESTSET,
                         checkpoint_type=eval_checkpoint_type,
                         fcam_argmax=False)

        if args.is_master:
            trainer.print_performances(checkpoint_type=eval_checkpoint_type)
            trainer.report(epoch, split=constants.TESTSET,
                           checkpoint_type=eval_checkpoint_type)
            trainer.save_performances(
                epoch=epoch, checkpoint_type=eval_checkpoint_type)

        trainer.switch_perf_meter_to_captured()

        DLLogger.log("EVAL time TESTSET - CHECKPOINT {}: {}".format(
            eval_checkpoint_type, dt.datetime.now() - t0))

    if args.distributed:
        dist.barrier()
    if args.is_master:
        trainer.save_args()
        trainer.plot_perfs_meter()
        bye(trainer.args)


if __name__ == '__main__':
    main()
