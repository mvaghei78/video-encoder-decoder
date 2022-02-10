# Video Encoder Decoder

in this project we implemented video encoder and video decoder.

main steps in encoder is:
1. apply DCT transformation on 8*8 blocks,
2. do quantization on 6*6 blocks,
3. apply zig-zag scan on 60*60 blocks,
4. apply run-length scan all over each,
5. and after doing previous steps on each frame we applied huffman coding on frames.

for decoding we implement reverse of encoding process.
so main steps of decoder is:
1. huffman decoding on output of encoder,
2. apply reverse of run-length scan,
3. apply reverse of zig-zag scan,
4. apply reverse of quantization step,
5. apply reverse of DCT transformation.

**notice**: *encoder.py* and *decoder.py* do there process on the video in gray scale, so if you want to keep video in RGB scale you can use *color_encoder.py* and *color_decoder.py*.

for running *encoder.py* and *color_encoder.py* you have to specify path of video with argument --video_path like this:

`python3 encoder.py --video_path ./a.avi`

**notice**: ecnoder output for *encoder.py* and *color_encoder.py* is stored in *./coded_frames/huffman_coded.txt* and *./colored_coded_frames/huffman_coded.txt*, respectively.

the output of steps 1 to 4 in encoding process contains numbers and spaces so huffman coding is stored its tree in *tree.txt* file that this file shows which character is the most frequent so the huffman coding could map that character to minimum number of bits. 

for decoding the encoded file you have to just running *decoder.py* and *color_decoder.py*.

**notice**: output of decoding process is stored in *./decoder_output.avi* for *decoder.py* and *./decoder_colored_output.avi* for *color_decoder.py*.

