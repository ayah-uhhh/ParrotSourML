"""tf trial"""
import logging
import time
import multiprocessing as mp
# from tdqm import tdqm
from PSLogger import psLog
import PSCNN as pscnn

psLog.setLevel(logging.DEBUG)


if __name__ == '__main__':
    # required conditional to avoid recursive multithreading

    psLog.info("Finding optimized parameters...")
    psLog.info("---------------")

    starttime = time.time()

    # initialize a thread pool equal to the number of cores available on
    # this machine
    pool = mp.Pool(mp.cpu_count(), maxtasksperchild=3)

    # Start with a error rate of 100%; if a RF instance beats this
    # they become the new best. If an img_size of -1 is the best result,
    # something went wrong
    best_accuracy = 0
    best_img_size = -1

    # Create threads for different optimizers, filters, kernel sizes and img_size values
    # this will find the best number within each range
    # def pscnn(optimizer='rmsprop', filters=3, kernel_size=(3, 3), img_size=100, show_chart=False, save=False, epochs=150, batch_size=32)
    optimizer = ['rmsprop', 'nadam', 'adam']
    for i in range(3):
        for j in range(3, 5):
            for l in range(80, 110, 10):
                results = [pool.apply_async(
                    pscnn.pscnn, args=([optimizer[i], j, (j, j), l, False, False, 150, 32, True]))]

    pool.close()

    # get the results after each thread has finished executing
    output = []
    for job in results:
        output.append(job.get())

    for x in output:
        if (x[3] > best_accuracy):
            best_accuracy = x[3]
            best_params = x[0]  # array
            best_optimizer = x[0][0]
            best_filters = x[0][1]
            best_kernel_size = x[0][2]
            best_img_size = x[0][3]
            model = x[4]

        if best_params is not None:
            with open('best_params.txt', 'w') as file:
                file.write(str(best_params))

        # only in logLevel DEBUG, print all results
        psLog.debug("-----------")
        psLog.debug("Img size: %s", str(x[0]))
        psLog.debug("Error: %s", str(x[2]))
        psLog.debug("")
        psLog.debug("Img size: %s", str(x[1]))

    psLog.debug("Saving best CNN model...")
    model.save('ps_cnn_model_2.h5')
    psLog.debug("Model saved.")
    psLog.info("------------------------------")

    total_time = time.time() - starttime

    # Results:
    psLog.info("Best error rate: %s", best_accuracy)
    psLog.info("Best img size: %s", best_img_size)
    psLog.info("Total time: %s", total_time)
