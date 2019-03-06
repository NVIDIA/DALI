import argparse
import os
import subprocess

default_format = "png"
default_qscale_jpg = "4"

def extract_frames(master_data, resolution, format, q, quiet,
                   transcoded, codec, crf, keyint):

    if transcoded:
        desc = [resolution, 'scenes']
        desc += [codec] if codec else []
        desc += ["crf"+crf] if crf else []
        desc += ["keyint"+keyint] if keyint else []
        in_path = os.path.join(master_data,*desc)
    else:
        if codec:
            raise ValueError("--codec specified, but not --transcoded");
        if crf:
            raise ValueError("--crf specified, but not --transcoded");
        if keyint:
            raise ValueError("--keyint specified, but not --transcoded");
        in_path = os.path.join(master_data,'orig','scenes')

    desc = [resolution,'frames']
    desc += [codec] if codec else []
    desc += ["crf"+crf] if crf else []
    desc += ["keyint"+keyint] if keyint else []
    if not format:
        format = default_format
    else:
        desc += [format]

    if not q:
        if format == "jpg":
            q = default_qscale_jpg
    else:
        desc += ["q" + q]

    out_path = os.path.join(master_data,*desc)

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
        res_args += ["-vf", "scale=%s" % res_str, "-sws_flags", "bilinear"]

    codec_args = []
    if format == "png":
        if q:
            codec_args += ["-compression_level", q]
    elif format == "jpg":
        codec_args += ["-q:v", q]
    else:
        raise ValueError("Unknown format")

    if quiet:
        cmdout = subprocess.DEVNULL
    else:
        cmdout = None

    for subset_name, subset_dir in [('training', 'train'), ('validation', 'val')]:
        if not os.path.exists(os.path.join(in_path,subset_dir)):
            raise ValueError("No "+subset_name+" data found in "+in_path+", " +
                             "did you run split_scenes.py?")

        for in_file in os.listdir(os.path.join(in_path,subset_dir)):
            if in_file.endswith('.mp4'):
                scene = in_file.split('_')[1].split('.')[0]
                cur_out_path = os.path.join(out_path,subset_dir,scene)
                if not os.path.isdir(cur_out_path):
                    os.makedirs(cur_out_path)
                cur_in_path = os.path.join(in_path,subset_dir,in_file)
                cmd = ["ffmpeg", "-n", "-i", cur_in_path]
                cmd += res_args
                cmd += codec_args
                cmd += [os.path.join(cur_out_path, "%05d."+format)]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, stdout=cmdout, stderr=cmdout)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_data', type=str, required=True,
                        help="Path to root data directory")
    parser.add_argument('--resolution', type=str, required=True,
                        choices=['4K', '1080p', '720p', '540p'])
    parser.add_argument('--format', type=str,
                        choices=['png', 'jpg'])
    parser.add_argument('-q', type=str,
                        help="quality to use for compression [2-31] for jpg and [0-9] for png")
    parser.add_argument('--transcoded', action='store_true',
                        help="Use transcoded videos instead of original split video")
    parser.add_argument('--quiet', action='store_true',
                        help="Suppress ffmpeg output")
    parser.add_argument('--codec', type=str, default=None,
                        choices=['h264', 'hevc'],
                        help="codec of transcoded video to use")
    parser.add_argument('--crf', type=str, default=None,
                        help="crf value of transcoded video to use")
    parser.add_argument('--keyint', type=str, default=None,
                        help="keyframe interval of transcoded video to use")
    args = parser.parse_args()
    extract_frames(**vars(args))
