import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali as dali
import os


def test_invariant_args():
    im = ndd.imread(os.path.join(os.environ["DALI_EXTRA_PATH"], "db", "imgproc", "alley.png"))

    size = 200

    with ndd.EvalContext() as ctx:
        mirror = False
        resized = ndd.resize_crop_mirror(
            im,
            crop=[224, 224],  # list of constants
            crop_pos_x=0.1 + 0.1,  # expression
            crop_pos_y=0.1,  # constant
            mirror=mirror,  # not a constant, because it's assigned more than once
            resize_shorter=size,  # local constant
            interp_type=dali.types.INTERP_LANCZOS3,
            antialias=False,
        )

        mirror = True  # makes `mirror` not invariant

        resized.evaluate()

        ops = list(ctx._instance_cache.items())
        assert len(ops) == 1
        call_args = set(ops[0][0][5])
        init_args = dict(ops[0][0][6])
        assert len(call_args) == 1
        assert "mirror" in call_args
        assert len(init_args) == 6
        assert "crop" in init_args
        assert "crop_pos_x" in init_args
        assert "crop_pos_y" in init_args
        assert "resize_shorter" in init_args
        assert "interp_type" in init_args
        assert "antialias" in init_args
