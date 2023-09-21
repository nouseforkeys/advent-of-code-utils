"""
example usage of advent of code utils
"""

from advent_of_code_utils import ParseConfig, parse_from_file

parser = ParseConfig('\n', ParseConfig(',', [int, float]))
output = parse_from_file('test_file.txt', parser)

print(output)
