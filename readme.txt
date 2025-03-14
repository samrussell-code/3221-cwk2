Complete the table below with your results, and then fill in the final two sections at the end.

Please do not edit or remove table or section headings, as the autograder script uses these to
locate the start and end of the table.

Each row of the table will be split into columns using the Python "split()" method. Therefore,
- fill in each column with values;
- only use whitespace between columns;
- do not add any non-numeric characters (which would cause the value to be read as zero);
- do not worry if your columns are not perfectly aligned.

For the parallel speed-up S, please note that:
- the time you should use is already output by the provided code;
- take as the serial execution time the time output by the code when run with a single process.
  Hence, the speed-up for 1 process in the table below must be 1.00.


No. Machines:   Total No. Processes:     Mean time (average of 3 runs) in seconds:        Parallel speed-up, S:
=============   ====================     =========================================        =====================
1                       1                          0.00982384                                        1                            
1                       2                          0.00702618                                        1.3982                                                                  
1                       4                          0.00659242                                        1.4902                                             
1                       8                          0.00457276                                        2.1483                            
2                       16                         0.0252841                                         0.3885                                                     
2                       32                         0.0613511                                         0.1601                                          

Please state the number of cores per machine (for Bragg 2.05, this is typically 12): 48

A brief interpretation of your results:

The speedup for a single machine gradually increases as number of total processes increases, since the workload is being split in parallel reducing the overall compute time.

Once the second machine is introduced, and the process count is doubled once again, the runtime actually increases to be longer than the series runtime.
I believe this is due to a combination of the network latency between the machines in the lab, and the initial overhead with distributing the function across processes.