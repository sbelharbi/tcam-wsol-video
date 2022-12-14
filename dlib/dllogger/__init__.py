# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# source: https://github.com/NVIDIA/dllogger

from .logger import (
    Backend,
    Verbosity,
    Logger,
    default_step_format,
    default_metric_format,
    StdOutBackend,
    JSONStreamBackend,
    ArbLogger,
    ArbStdOutBackend,
    ArbTextStreamBackend,
    ArbJSONStreamBackend
)

__version__ = "0.1.0"


class DLLoggerNotInitialized(Exception):
    pass


class DLLLoggerAlreadyInitialized(Exception):
    pass


class NotInitializedObject(object):
    def __getattribute__(self, name):
        raise DLLoggerNotInitialized(
            "DLLogger not initialized. "
            "Initialize DLLogger with init(backends) function"
        )


GLOBAL_LOGGER = NotInitializedObject()


def log(step, data, verbosity=Verbosity.DEFAULT):
    GLOBAL_LOGGER.log(step, data, verbosity=verbosity)


def log(message, verbosity=Verbosity.DEFAULT):
    GLOBAL_LOGGER.log(message, verbosity=verbosity)


def metadata(metric, metadata, is_master: bool):
    GLOBAL_LOGGER.metadata(metric, metadata, is_master)


def flush():
    GLOBAL_LOGGER.flush()


def init(backends, master_pid: int):
    global GLOBAL_LOGGER
    try:
        if isinstance(GLOBAL_LOGGER, Logger):
            raise DLLLoggerAlreadyInitialized()
    except DLLoggerNotInitialized:
        GLOBAL_LOGGER = Logger(backends, master_pid=master_pid)


def init_arb(backends, is_master: bool, reset: bool = False,
             flush_at_log: bool = True):
    global GLOBAL_LOGGER
    try:
        if isinstance(GLOBAL_LOGGER, ArbLogger) and not reset:
            raise DLLLoggerAlreadyInitialized()
        elif isinstance(GLOBAL_LOGGER, ArbLogger) and reset:
            GLOBAL_LOGGER.flush()
            GLOBAL_LOGGER = ArbLogger(backends, is_master=is_master,
                                      flush_at_log=flush_at_log)
        elif not isinstance(GLOBAL_LOGGER, ArbLogger) and reset:
            GLOBAL_LOGGER = ArbLogger(backends, is_master=is_master,
                                      flush_at_log=flush_at_log)
    except DLLoggerNotInitialized:
        GLOBAL_LOGGER = ArbLogger(backends, is_master=is_master,
                                  flush_at_log=flush_at_log)
