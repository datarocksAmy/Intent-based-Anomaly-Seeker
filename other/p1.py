"""
p1

Write a function called x_metric.

The x_metric takes a list of non-negative integers as input. This list is called x_list.

The x_metric should output the x score, which is the largest number x such that x elements
in x_list are greater or equal to x.

-------------------
x_metric([1,2,3,4,5])
>> 3
x_metric([1,1,1,1,1])
>> 1
x_metric([5,5,5,5,5])
>> 5

"""

from collections import Counter

def x_metric(x_list: "list of non-negative ints") -> int:
    """
    Find the max number of the list.
    Return that max number only when count(numbers in list >= max number) = max_num.

    :param x_list:  ( List ) List of non-negative integers
    :return:        ( Int ) max_num of the list & max_num = count(elements >= max_num)
    """

    # Sort input list in descending order
    x_list.sort()
    # Count occurrence of each element to find duplications
    x_frequency = Counter(x_list)
    # Go through elements, starting from the biggest number
    for element in x_list:
        # Boolean indicator for value >= current iteration element. Count number of True
        num_greater_equal_element = list(x >= element for x in x_list).count(True)
        # Check if the numbers in the list is unique
        if x_frequency[max(x_frequency)] > 1:
            # Assign max value in that list as init x_max_frequency
            x_max_frequency = max(x_frequency)
            # Get the frequency value of the max number
            x_frequency_count = x_frequency[x_max_frequency]
            # For only 1 unique number in the list, get the max number
            if [x_max_frequency == x for x in x_list].count(True) == len(x_list):
                return x_max_frequency
            # For mixed type of numbers with duplicates, get the frequency count
            else:
                return x_frequency_count
        # Unique numbers list : Largest number x >= count
        elif element >= num_greater_equal_element:
            return element



# Test
# input_l = [1, 2, 3, 4, 5]     # 3
# input_l = [1, 1, 1, 1, 1]     # 1
# input_l = [5, 5, 5, 5, 5]     # 5
# input_l = [1, 3, 1, 3, 1]     # 2
# input_l = [5, 5, 5, 5, 1]     # 4
# input_l = [3, 3, 3, 3, 3]     # 3
input_l = [3, 1, 3, 1, 3]       # 3
output = x_metric(input_l)
print(f"OUTPUT : {output}")