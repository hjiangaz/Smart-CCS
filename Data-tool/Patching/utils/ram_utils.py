from multiprocessing import Process
import shutil
import uuid
import multiprocessing
import os
import fcntl
import json
import stat
import random
import time
import json


def __read_file(p):
    with open(p, 'rb') as f:
       _ = f.read()
    print('Loaded:', p)


def hit_cache(path):
    p = multiprocessing.Process(target=__read_file, args=(path,))
    p.start()


def lock(f):
    fcntl.flock(f, fcntl.LOCK_EX)

 
def unlock(f):
    fcntl.fcntl(f, fcntl.LOCK_UN)


class RamDiskCache:
    cfg_name='use.json'

    def __init__(self, root, capacity, warning_mem=5) -> None:
        """
        root: the cache root
        capacity: the size of cache file, GB
        warning_mem: stop caching if the left memory is smaller than warning_mem (GB)
        """
        self.root = root
        self.capacity = capacity*1024   #MB
        self.warning_mem = warning_mem*1024   #MB
        self.used_mem = 0
        self.maintain = {}
        self.init()

    def init(self):
        p = os.path.join(self.root, self.cfg_name)
        print('Creating config file at:', p)
        if not os.path.exists(p):
            self.__write_cfg()
        else:
            self.__read_cfg()
            self.__clean()
            self.__write_cfg()
            self.used_mem = self.__cal_used_mem()
    
    def __clean(self):
        temp = {}
        for src, target in self.maintain.items():
            if os.path.exists(target):
                temp[src] = target
                print('{} is not exists, updating cfg file'.format(target), flush=True)
        self.maintain = temp

    def add_cache(self, path, join=False):
        self.__read_cfg()
        left_mem = self.capacity - self.used_mem
        while left_mem < self.warning_mem:
            src = random.choice(self.maintain.keys())
            print('Left space: {} MB, removing {}'.format(left_mem, self.maintain[src]))
            self.remove_cache(src)
            self.used_mem = self.__cal_used_mem()
            left_mem = self.capacity - self.used_mem

        if isinstance(path, str):
            path = [path]
        for p in path:
            if p in self.maintain.keys():
                print(p, ': is already in the cache list')
            else:
                process = Process(target=self.__copy_file, args=(p,))
                process.start()
                if join:
                    process.join()
    
    def remove_cache(self, path):
        self.__read_cfg()

        if isinstance(path, str):
            path = [path]
        path = [p for p in path if p in self.maintain.keys()]
        for p in path:
            process = Process(target=self.__remove_file, args=(p, ))
            process.start()

    def new_path(self, src):
        self.__read_cfg()

        if src in self.maintain.keys():
            target = self.maintain[src]
        else:
            target = src
        return target

    def __copy_file(self, src):
        # generate temp file name
        postfix = src.split('.')[-1]
        name = str(uuid.uuid3(namespace=uuid.NAMESPACE_DNS, name=src)) + '.' + postfix
        target = os.path.join(self.root, name)
        try:
            s = time.time()
            shutil.copyfile(src=src, dst=target, follow_symlinks=False)
            print('Cache time:', time.time() - s, flush=True)
            self.increase_cfg(src=src, target=target)
        except Exception as e:
            print('Failed to cache file', e)
    
    def __remove_file(self, src):
        target = self.maintain[src]
        os.remove(target)
        self.decrease_cfg(src)
        print('remove successfuly:', target)
    
    def __cal_used_mem(self):
        used_mem = 0
        self.__read_cfg()
        for _, v in self.maintain.items():
            size = int(os.path.getsize(v)/1024/1024) #MB
            used_mem += size
        return used_mem
    
    def __read_cfg(self) -> dict:
        p = os.path.join(self.root, self.cfg_name)
        with open(p) as f:
            lock(f)
            self.maintain = json.load(f)
            unlock(f)

    def __write_cfg(self):
        p = os.path.join(self.root, self.cfg_name)
        with open(p, 'w') as f:
            lock(f)
            json.dump(self.maintain, f)
            unlock(f)
        rights = oct(os.stat(p).st_mode)[-3:]
        if rights != '777':
            os.chmod(p, mode=stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            rights = oct(os.stat(p).st_mode)[-3:]
            print('{} mode is: {}'.format(p, rights))

    def increase_cfg(self, src, target):
        # load cfg, other process may update it 
        self.__read_cfg()
        # update maintain list
        self.maintain[src] = target
        # write back the maintain list 
        self.__write_cfg()
        # update memory 
        self.used_mem = self.__cal_used_mem()
        print('Sucessfully cached: {}, used storage: {:.2f} GB, left space: {:.2f} GB'.format(
            src, self.used_mem/1024, (self.capacity - self.used_mem)/1024
        ), flush=True)

    def decrease_cfg(self, src):
        # load cfg, other process may update it
        self.__read_cfg()
        # update maintain list
        try:
            t = self.maintain.pop(src)
        except:
            print('{} does not exist ...'.format(src))
        # write back the maintain list 
        self.__write_cfg()
        # update memory 
        self.used_mem = self.__cal_used_mem()
        print('Sucessfully cached: {}, used storage: {:.2f} GB, left space: {:.2f} GB'.format(
            src, self.used_mem/1024, (self.capacity - self.used_mem)/1024
        ))


if __name__ == '__main__':
    import time
    # step 1, create ramdisk use following cmd, ask help from your admin. Of course, you can also use a SSD path as buffer
    # sudo mount -t tmpfs -o rw,size=100G tmpfs /mnt/ramdisk
    # Step 2, use
    def dummy_work_for_test(p):
        time.sleep(2)
        print(p, ':is finished...')

    json_path = '/jhcnas3/backup/scripts/datasets/cervical_dataset.json'
    paths = []
    json_file = json.load(open(json_path))
    images = json_file['images']
    for image in images:
            paths.append(image['original_path'])
            
    cache_engine = RamDiskCache('/tmp/ramdisk', 100)

    for index in range(len(paths)):

        curr_path = paths[index]
        print('current:', curr_path)
        next_index = (index + 1) % len(paths)
        next_path = paths[next_index]

        # background cache
        cache_engine.add_cache(next_path)   # if you want to cache more, pass it a list such as [path1, path2]

        # find new path, if the path is alread cached, then it will return the cached path, else return the input
        cache_path = cache_engine.new_path(curr_path)
        # current work
        dummy_work_for_test(cache_path)
        # remove cached file
        cache_engine.remove_cache(curr_path) # manually remove cached file, if you forget this, I will randomly pop used items if the memory is not enough.
