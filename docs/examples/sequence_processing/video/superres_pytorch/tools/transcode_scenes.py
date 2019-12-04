import argparse
import os
import subprocess

default_codec = "h264"
default_crf = "18"
default_keyint = "4"

def downsample_scenes(master_data, resolution, codec, crf, keyint, quiet):

    desc = [resolution, 'scenes']
    if not codec:
        codec = default_codec
    else:
        desc += [codec]
    assert codec in ['h264', 'hevc'], '--codec must be one of h264 or hevc'

    if not crf:
        crf = default_crf
    else:
        desc += ["crf" + crf]

    if not keyint:
        keyint = default_keyint
    else:
        desc += ["keyint" + keyint]

    master_out_path = os.path.join(master_data,*desc)

    print("Writing output files to:", master_out_path)

    for subset in ['train', 'val']:
        if not os.path.isdir(os.path.join(master_out_path,subset)):
            os.makedirs(os.path.join(master_out_path,subset))

    res_args = []
    if resolution == '4K':
        pass
    else:
        if resolution == '1080p':
            res_str = '1920:1080'
        elif resolution == '720p':
            res_str = '1280:720'
        elif resolution == '540p':
            res_str = '960:540'
        else:
            raise ValueError("Unknown resolution")
        res_args = ["-vf", "scale=%s" % res_str, "-sws_flags", "bilinear"]

    codec_args = ["-preset", "slow"]
    if codec == 'h264':
        codec_args = ["-c:v", "libx264", "-g", keyint,
                      "-profile:v", "high"]
    elif codec == 'hevc' or codec == 'h265':
        codec_args = ["-c:v", "libx265", "-x265-params",
                      "keyint=%s:no-open-gop=1" % (keyint)]
    else:
        raise ValueError("Unknown codec")

    if quiet:
        cmdout = subprocess.DEVNULL
    else:
        cmdout = None

    def transcode(in_path, out_path):
        cmd = ["ffmpeg", "-y", "-i", in_path]
        cmd += res_args
        cmd += codec_args
        cmd += ["-crf", crf, "-an", out_path]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, stdout=cmdout, stderr=cmdout)

    for subset in ['train', 'val']:
        for in_file in os.listdir(os.path.join(master_data,'orig','scenes',subset)):
            if in_file.endswith('.mp4'):
                in_path = os.path.join(master_data,'orig','scenes',subset,in_file)
                out_path = os.path.join(master_out_path,subset,in_file)
                transcode(in_path, out_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_data', type=str, default=None,
                        help="Path to root data directory")
    parser.add_argument('--resolution', type=str, default=None,
                        help="one of '4K', '1080p', '720p', or '540p'")
    parser.add_argument('--codec', type=str, default=None,
                        help="one of 'h264' or 'hevc'")
    parser.add_argument('--crf', type=str, default=None,
                        help="crf value passed to ffmpeg")
    parser.add_argument('--keyint', type=str, default=None,
                        help="keyframe interval")
    parser.add_argument('--quiet', action='store_true',
                        help="Suppress ffmpeg output")
    args = parser.parse_args()
    assert args.master_data is not None, 'Provide --master_data path to root data directory containing split scenes'
    assert args.resolution in ['4K', '1080p', '720p', '540p'], '--resolution must be one of 1080p, 720p, 540p'
    downsample_scenes(**vars(args))
