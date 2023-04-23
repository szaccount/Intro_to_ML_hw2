#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
    # HELP FUNCTIONS MOVE
    def prob_given_x(self, x):
        """
        Returns the probabilities for y = 1 given x, y = 0 given x (based on P) for the passed x.
        """
        if (0 <= x <= 0.2) or (0.4 <= x <= 0.6) or (0.8 <= x <= 1):
            p = 0.8
        else:
            p = 0.1
        return [p, 1-p]
    
    def interval_lists_intersection(self, intervals1, intervals2):
        """
        Input: two lists of intervals.

        Returns: the intervals of the intersection between the two lists.
        """
        intersection = []
        index1 = 0
        index2 = 0

        while index1 < len(intervals1) and index2 < len(intervals2):
            # Checking if intervals1[index1] intersects intervals2[index2]
            start = max(intervals1[index1][0], intervals2[index2][0])
            end = min(intervals1[index1][1], intervals2[index2][1])
            if start <= end:
                intersection.append([start, end])

            # Skip the interval with the lowest endpoint
            if intervals1[index1][1] < intervals2[index2][1]:
                index1 += 1
            else:
                index2 += 1

        return intersection

    # DELETE THIS
    def calculate_len_intersection(self, list1, list2):
        """Calculate the length of the intersection between the intervals in list1 and list2.
        Input: list1 - a list of tuples, every tuple is an interval.
            list2 - a list of tuples, every tuple is an interval.

        Returns: The length of the intersection between the intervals in list1 and list2.
        """
        len_intersection = 0
        pointer_list1 = 0
        pointer_list2 = 0
        while pointer_list1 < len(list1) and pointer_list2 < len(list2):
            start = max(list1[pointer_list1][0], list2[pointer_list2][0])
            end = min(list1[pointer_list1][1], list2[pointer_list2][1])
            if start < end:
                len_intersection += (end - start)
            if list1[pointer_list1][1] == list2[pointer_list2][1]:
                pointer_list1 += 1
                pointer_list2 += 1
            elif list1[pointer_list1][1] < list2[pointer_list2][1]:
                pointer_list1 += 1
            else:
                pointer_list2 += 1
        return len_intersection
    # DELETE THIS

    def true_error_for_intervals(self, intervals):
        """
        Returns the true error with resepect to the distribution P and the passed intervals (define hypothesis).
        """
        # accumulates area where passed intervals are in [0,0.2]U[0.4,0.6]U[0.8,1]
        hypothesis_1_high_prob_area = 0
        # accumulates area where passed intervals are in [0.2,0.4]U[0.6,0.8]
        hypothesis_1_low_prob_area = 0
        intersection_1_high_prob = self.interval_lists_intersection([[0,0.2], [0.4,0.6], [0.8,1]], intervals)
        intersection_1_low_prob = self.interval_lists_intersection([[0.2,0.4], [0.6,0.8]], intervals)
        for interval in intersection_1_high_prob:
            hypothesis_1_high_prob_area += (interval[1] - interval[0])
        for interval in intersection_1_low_prob:
            hypothesis_1_low_prob_area += (interval[1] - interval[0])
        
        hypothesis_0_high_prob_area = 0.4 - hypothesis_1_low_prob_area
        hypothesis_0_low_prob_area = 0.6 - hypothesis_1_high_prob_area

        # real gives 1 while hypothesis 0
        expect_real_1_hypo_0 = (0.8 * hypothesis_0_low_prob_area) + (0.1 * hypothesis_0_high_prob_area)
        expect_real_0_hypo_1 = (0.2 * hypothesis_1_high_prob_area) + (0.9 * hypothesis_1_low_prob_area)
        return expect_real_1_hypo_0 + expect_real_0_hypo_1

    def sample_x_from_D(self, m):
        """Samples m samples of x from the distribution.
        Input: m - an integer, the size of the data sample.

        Returns: array of the sampled x values.
        """
        left = 0
        right = 1
        return np.random.uniform(left,right,m)
    
    def tags_of_sampled_x(self, x_samples):
        """Samples the tags of the passed x samples.
        Input: x_samples, the sampled x values.

        Returns: array of tags sampled for the x samples. 
        """
        tags = [1,0]
        return [np.random.choice(tags, p=self.prob_given_x(x)) for x in x_samples]
    
    def error_value_of_hypothesis_on_sample(self, intervals, x_value, y_value):
        """Calculates the error value (0 or 1) of the hypothesis on the sample.
        Input:
        intervals - the hypothesis.
        x_value - the x value of the sample.
        y_value - the y value of the sample (real tag).

        Returns: Error value (0 or 1) of the hypothesis on the sample (0 if hypothesis correct).
        """
        # Checking if the sample is in the intervals to have the classification of the hypothesis
        contained = False
        for interval in intervals:
            if interval[0] <= x_value <= interval[1]:
                contained = True
                break
        if (contained and y_value == 1) or (not contained and y_value == 0):
            return 0
        else:
            return 1

    def empirical_error_for_hypothesis(self, intervals, x_samples, y_samples):
        """Calculates the empirical error for the passed hypothesis (intervals) on the passed samples.
        Input:
        intervals - the hypothesis.
        x_samples - x values of the samples.
        y_samples - y values of the samples.

        Returns: Empirical error of the hypothesis on the samples.
        """
        num_errors = 0
        num_samples = len(x_samples)
        for i in range(num_samples):
            num_errors += self.error_value_of_hypothesis_on_sample(intervals, x_samples[i], y_samples[i])
        
        return num_errors / num_samples

    # HELP FUNCTIONS MOVE

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_sample = self.sample_x_from_D(m)
        x_sample.sort()
        y_sample = self.tags_of_sampled_x(x_sample)
        pairs = []
        for i in range(len(x_sample)):
            pairs.append([x_sample[i], y_sample[i]])
        return np.array(pairs)


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        ret_val = []
        empirical_err_averages = []
        true_err_averages = []
        sample_sizes = list(range(m_first, m_last + 1, step))
        for n in sample_sizes:
            empirical_err_sum = 0
            true_err_sum = 0
            for experiment in range(T):
                samples = self.sample_from_D(n)
                samples_x = [sample[0] for sample in samples]
                samples_y = [sample[1] for sample in samples]
                erm = intervals.find_best_interval(samples_x, samples_y, k)
                erm_intervals, error_count = erm
                empirical_err = error_count / n
                true_err = self.true_error_for_intervals(erm_intervals)

                empirical_err_sum += empirical_err
                true_err_sum += true_err
            
            empirical_err_avg = empirical_err_sum / T
            true_err_avg = true_err_sum / T
            ret_val.append([empirical_err_avg, true_err_avg])
            empirical_err_averages.append(empirical_err_avg)
            true_err_averages.append(true_err_avg)
        
        plt.figure(1)
        plt.title("Empirical and true errors averaged as a function of n (sample size)")
        plt.xlabel("n (Sample size)")
        plt.ylabel("Error")
        plt.plot(sample_sizes, empirical_err_averages, '.', label="empirical error")
        plt.plot(sample_sizes, true_err_averages, '.', label="true error")
        plt.legend()

        return np.array(ret_val)



    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        empirical_errors = []
        true_errors = []
        k_vals = list(range(k_first, k_last + 1, step))
        k_star = -1
        k_star_err = 10
        samples = self.sample_from_D(m)
        samples_x = [sample[0] for sample in samples]
        samples_y = [sample[1] for sample in samples]
        for k in k_vals:
            erm_intervals, error_count = intervals.find_best_interval(samples_x, samples_y, k)
            empirical_error = error_count / m
            true_error = self.true_error_for_intervals(erm_intervals)
            empirical_errors.append(empirical_error)
            true_errors.append(true_error)
            if k_star_err > empirical_error:
                k_star = k
                k_star_err = empirical_error
        
        plt.figure(2)
        plt.title("Empirical and true errors as a function of k")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.plot(k_vals, empirical_errors, '.', label="empirical error")
        plt.plot(k_vals, true_errors, '.', label="true error")
        plt.legend()

        plt.show()

        return k_star


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        samples_x = self.sample_x_from_D(m)
        separator = int(0.8 * m)
        x_train = samples_x[:separator]
        x_test = samples_x[separator:]
        x_train.sort()
        x_test.sort()
        y_train = self.tags_of_sampled_x(x_train)
        y_test = self.tags_of_sampled_x(x_test)

        best_k = -1
        best_empirical_error = 10
        # It was said in the forum to only draw the samples once
        for k in range(1,11):
            erm_intervals, error_count_train = intervals.find_best_interval(x_train, y_train, k)
            empirical_error_test = self.empirical_error_for_hypothesis(erm_intervals, x_test, y_test)
            if empirical_error_test < best_empirical_error:
                best_empirical_error = empirical_error_test
                best_k = k
        
        print(f"{best_k=}, {best_empirical_error=}")
        return best_k

        

    #################################
    # Place for additional methods


    #################################


if __name__ == '__main__':
    ass = Assignment2()
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)

