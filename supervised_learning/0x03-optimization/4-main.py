#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
moving_average = __import__('4-moving_average').moving_average

if __name__ == '__main__':
        data = [72, 78, 71, 68, 66, 69, 79, 79, 65, 64, 66, 78, 64, 64, 81, 71, 69,
                65, 72, 64, 60, 61, 62, 66, 72, 72, 67, 67, 67, 68, 75]
        days = list(range(1, len(data) + 1))
        m_avg = moving_average(data, 0.9)
        print(m_avg)
        m_avg2 = moving_average(data, 0.98)
        print(m_avg2)
        m_avg3 = moving_average(data, 0.5)
        print(m_avg3)
        plt.plot(days, data, 'r', days, m_avg, 'b',
		 days, m_avg2, 'y', days, m_avg3, 'g')
        plt.xlabel('Day of Month')
        plt.ylabel('Temperature (Fahrenheit)')
        plt.title('SF Maximum Temperatures in October 2018')
        plt.legend(['actual', 'moving_average beta=0.9',
		    'moving_average beta=0.98', 'moving_average beta=0.5'])
        plt.show()
