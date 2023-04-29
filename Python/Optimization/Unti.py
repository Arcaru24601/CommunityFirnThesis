from multiprocessing import Process
import sys
import time

def func_once():
    print("this func works one.")

func_once()
def func1():
    rocket = 0
    print ('start func1')
    while rocket < 5000000000:
        rocket += 1
    print ('end func1')

def func2():
    rocket = 0
    print ('start func2')
    while rocket < 5000000000:
        rocket += 1
    print ('end func2')

if __name__=='__main__':
    t1 = time.time()
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()
    p1.join()
    p2.join()
    print(time.time()-t1,'Parallel')
    t2 = time.time()
    func1()
    print(time.time()-t2,'Serial 1')
    t3 = time.time()
    func2()
    print(time.time()-t3,'Serial 2')
