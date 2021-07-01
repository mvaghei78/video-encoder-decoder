import json
import os


class Huffman:

    def __init__(self, address,tree_address=None):
        self.codes = {}
        if tree_address is None:
            self.string_input=open("coded_frames/encoded_video.txt").readline()
            freqs = self.frequency(self.string_input)
            self.tuples = self.sort_freq(freqs)
        else:
            self.encoded_output=self.read_binary_file(address)
            self.tuples=self.read_json(tree_address)
        tree=self.build_tree(self.tuples)
        self.trim=self.trim_tree(tree)


    @staticmethod
    def frequency(string_input):
        freq = {}
        for ch in string_input:
            freq[ch] = freq.get(ch, 0) + 1
        return freq

    @staticmethod
    def sort_freq(freq):
        letters = freq.keys()
        output_tuples = []
        for let in letters:
            output_tuples.append((freq[let], let))
        output_tuples.sort()
        return output_tuples

    @staticmethod
    def build_tree(input_tuples):
        while len(input_tuples) > 1:
            least_two = tuple(input_tuples[0:2])  # get the 2 to combine
            the_rest = input_tuples[2:]  # all the others
            comb_freq = least_two[0][0] + least_two[1][0]  # the branch points freq
            input_tuples = the_rest + [(comb_freq, least_two)]  # add branch point to the end
            input_tuples.sort()  # sort it into place
        return input_tuples[0]  # Return the single tree inside the list

    @classmethod
    def trim_tree(cls, tree):
        # Trim the freq counters off, leaving just the letters
        p = tree[1]  # ignore freq count in [0]
        if type(p) == type(""):
            return p  # if just a leaf, return it
        else:
            return cls.trim_tree(p[0]), cls.trim_tree(p[1])

    def assign_codes(self, node, pat=''):
        if type(node) == type(""):
            self.codes[node] = pat  # A leaf. set its code
        else:  #
            self.assign_codes(node[0], pat + "0")  # Branch point. Do the left branch
            self.assign_codes(node[1], pat + "1")  # then do the right branch.

    def encode(self):
        self.assign_codes(self.trim)
        str=self.string_input
        output = ""
        for ch in str: output += self.codes[ch]
        self.write_binary_file("coded_frames/huffman_coded.txt",output)
        self.write_tuple_to_json(self.tuples)

    def decode(self):
        tree=self.trim
        str=self.encoded_output
        decoded = ""
        p = tree
        for bit in str:
            if bit == '0':
                p = p[0]  # Head up the left branch
            else:
                p = p[1]  # or up the right branch
            if type(p) == type(""):
                decoded += p  # found a character. Add to output
                p = tree  # and restart for next character
        return decoded

    @staticmethod
    def read_binary_file(address):
        with open(address, "rb") as f:
            output_in_number = list(f.read())
            f.close()
            string = ""
            for i in output_in_number:
                string += (format(i, '08b'))
            return string

    @staticmethod
    def write_binary_file(address, output):

        bit_strings = [output[i:i + 8] for i in range(0, len(output), 8)]

        # then convert to integers
        byte_list = [int(b, 2) for b in bit_strings]
        bit_arr = bytearray(byte_list)
        with open(address, 'wb') as binary_file:
            binary_file.write(bit_arr)
            binary_file.close()

    @classmethod
    def write_tuple_to_json(cls,input_tuple):
        json_out = {}
        for row in input_tuple:
            json_out[row[1]] = row[0]
        cls.save_json(json_out,"coded_frames/tree.txt")
    @staticmethod
    def save_json(dict,address):

        with open(address, 'w') as outfile:
            json.dump(dict, outfile)
    @staticmethod
    def read_json(address):
        with open(address) as json_file:
            data = json.load(json_file)
            return [(v, k) for k, v in data.items()]

# huffman=Huffman("coded_frames/huffman_coded.txt","coded_frames/tree.txt")
# x=huffman.decode()
# print ("a")
# codes = {}
# input_string = open(os.path.join("coded_frames", "encoded_video.txt")).readline()
#
# freqs = frequency(input_string)
# tuples = sortFreq(freqs)
# tree = buildTree(tuples)
# trim = trimTree(tree)
# assignCodes(trim)
# output = encode(input_string)
# if len(output) % 8 != 0:
#     m = ""
#     for i in range(8 - (len(output) % 8)):
#         m = m + "0"
#     output = m + output
# bit_strings = [output[i:i + 8] for i in range(0, len(output), 8)]
#
# # then convert to integers
# byte_list = [int(b, 2) for b in bit_strings]
# bit_arr = bytearray(byte_list)
# with open('byte.bin', 'wb') as f:
#     f.write(bit_arr)
#     f.close()
# with open("byte.bin", "rb") as f:
#     number = list(f.read())
#     f.close()
# string = ""
# for i in number:
#     string += (format(i, '08b'))
# input = decode(trim, string)
