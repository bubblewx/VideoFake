import os
import sys
import importlib
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import roop

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_frames',
    'process_image',
    'process_video',
    'post_process'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'roop.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError:
        sys.exit(f'Frame processor {frame_processor} not found.')
    except NotImplementedError:
        sys.exit(f'Frame processor {frame_processor} not implemented correctly.')
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    return FRAME_PROCESSORS_MODULES


def list_processor_files(processor_name: str) -> List[str]:
    """返回以 processor_name 开头的文件名列表"""
    process_dir = "/content/drive/MyDrive/process/"

    if not os.path.exists(process_dir):
        print(f"Warning: Directory {process_dir} does not exist.")
        return []


    processor_files = [
        entry.name[len(processor_name):] for entry in os.scandir(process_dir)
        if entry.is_file() and entry.name.startswith(processor_name)
    ]

    print("Processed files:", processor_files)  # 打印文件列表
    return processor_files


def create_filtered_queue(temp_frame_paths: List[str], processor_name: str) -> Queue:
    """创建一个过滤后的队列，排除已处理的文件"""
    processor_files = list_processor_files(processor_name)
    queue = Queue()

    for path in temp_frame_paths:
        # 获取文件名
        file_name = os.path.basename(path)

        # 判断文件是否已处理
        if file_name not in processor_files:
            queue.put(path)
        else:
            print(f"{file_name} already executed, ignoring.")

    return queue

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], update: Callable[[], None], processor_name: str) -> None:
    with ThreadPoolExecutor(max_workers=roop.globals.execution_threads) as executor:
        futures = []
        queue = create_filtered_queue(temp_frame_paths, processor_name)
        queue_per_future = max(len(temp_frame_paths) // roop.globals.execution_threads, 1)
        while not queue.empty():
            future = executor.submit(process_frames, source_path, pick_queue(queue, queue_per_future), update)
            futures.append(future)
        for future in as_completed(futures):
            future.result()


def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None], processor_name: str) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        multi_process_frame(source_path, frame_paths, process_frames, lambda: update_progress(progress), processor_name)


def update_progress(progress: Any = None) -> None:
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
    progress.set_postfix({
        'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
        'execution_providers': roop.globals.execution_providers,
        'execution_threads': roop.globals.execution_threads
    })
    progress.refresh()
    progress.update(1)
