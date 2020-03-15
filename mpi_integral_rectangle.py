# -*- coding: utf-8 -*
import sys

import numpy
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if comm.rank == 0:
    fout = open('res.txt', 'w')

def f(x):
    return 14 * x / (10 + x * x)

def Integrate_Rectangle(a, b, n):
    h = (b - a) / n
    integral = 0
    for i in range(1, int(n)):
        x = a + i * h
        integral += f((x - h + x) * 0.5) * h
    return integral

def main(ga, gb, gn):
    h = (gb - ga) / gn
    ln = gn / size
    la = ga + rank * ln * h
    lb = la + ln * h

    integral = numpy.zeros(1)
    buf = numpy.zeros(1)
    
    tn = MPI.Wtime()
    
    integral[0] = Integrate_Rectangle(la, lb, ln)

    if rank == 0:
        total = integral[0]
        for i in range(1, size):
            comm.Recv(buf, ANY_SOURCE)
            total += buf[0]
    else:
        comm.Send(integral, 0)

    tk = MPI.Wtime()
    
    if comm.rank == 0:
        fout.write("При n = " + str(gn) + " подинтегралов, общая сумма интеграла от " + str(ga) + " до " + str(gb) + " равна " + str(total))
        fout.write("\n")
        fout.write("Время выполнения программы на " + str(size) + " процессах равно " + str(tk - tn) + " секунд")
        fout.close()
    return 0

if __name__ == '__main__':
    ga = float(sys.argv[1])
    gb = float(sys.argv[2])
    gn = int(sys.argv[3])
    main(ga, gb, gn)
    
