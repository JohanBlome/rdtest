#!/usr/bin/env python3

"""rdtest.py module description."""

# Invalid name "get_info" (should match ^_?[A-Z][a-zA-Z0-9]*$)
# pylint: disable-msg=C0103

import argparse
import itertools
import os
import pandas as pd
import pathlib
import sys
import re

import utils


default_values = {
    "debug": 0,
    "cleanup": 0,
    "ref_res": None,
    "ref_pix_fmt": "yuv420p",
    "vmaf_dir": "/tmp/",
    "tmp_dir": "/tmp/",
    "infile_list": [],
    "outfile": None,
}


def run_experiment(options):
    # check all software is ok
    utils.check_software(options.debug)

    # prepare output directory
    pathlib.Path(options.tmp_dir).mkdir(parents=True, exist_ok=True)

    df = None

    columns = (
        "infile",
        "codec",
        "resolution",
        "width",
        "height",
        "framerate",
        "rcmode",
        "quality",
        "preset",
        "encoder_duration",
        "actual_bitrate",
        "psnr_y_mean",
        "psnr_u_mean",
        "psnr_v_mean",
        "psnr_y_p0",
        "psnr_y_p5",
        "psnr_y_p10",
        "psnr_y_p25",
        "psnr_y_p75",
        "psnr_y_p90",
        "psnr_y_p95",
        "psnr_y_p100",
        "psnr_u_p0",
        "psnr_u_p5",
        "psnr_u_p10",
        "psnr_u_p25",
        "psnr_u_p75",
        "psnr_u_p90",
        "psnr_u_p95",
        "psnr_u_p100",
        "psnr_v_p0",
        "psnr_v_p5",
        "psnr_v_p10",
        "psnr_v_p25",
        "psnr_v_p75",
        "psnr_v_p90",
        "psnr_v_p95",
        "psnr_v_p100",
        "ssim_y_mean",
        "ssim_u_mean",
        "ssim_v_mean",
        "ssim_y_p0",
        "ssim_y_p5",
        "ssim_y_p10",
        "ssim_y_p25",
        "ssim_y_p75",
        "ssim_y_p90",
        "ssim_y_p95",
        "ssim_y_p100",
        "ssim_u_p0",
        "ssim_u_p5",
        "ssim_u_p10",
        "ssim_u_p25",
        "ssim_u_p75",
        "ssim_u_p90",
        "ssim_u_p95",
        "ssim_u_p100",
        "ssim_v_p0",
        "ssim_v_p5",
        "ssim_v_p10",
        "ssim_v_p25",
        "ssim_v_p75",
        "ssim_v_p90",
        "ssim_v_p95",
        "ssim_v_p100",
        "vmaf_mean",
        "vmaf_harmonic_mean",
        "vmaf_p0",
        "vmaf_p5",
        "vmaf_p10",
        "vmaf_p25",
        "vmaf_p75",
        "vmaf_p90",
        "vmaf_p95",
        "vmaf_p100",
        "parameters",
    )
    df = pd.DataFrame(columns=columns)
    for infile in options.infile_list:
        reg = r"^(?P<source>[\w.]*).y4m"
        match = re.search(reg, infile)
        if match:
            source = match.group("source") + ".y4m"
            rest = infile[len(source) + 1 :]
            match = re.findall(r"(?P<width>[0-9]*)x(?P<height>[0-9]*)", source)
            # should take the last but -1 does not work
            width = match[-1][0]
            height = match[-1][1]
            resolution = f"{width}x{height}"

            match = re.search(f"qp(?P<qp>[0-9]*)", rest)
            qp = match.group("qp")
            match = re.findall(r"(?P<fps>[0-9.]*)fps", source)
            fps = match[-1]
            ref_path = f"media/{source}"
            codec = utils.get_codec_name(infile)
            duration = utils.get_duration(infile)
            actual_bitrate = utils.get_bitrate(infile)
            ref_framerate = fps
            quality = qp
            preset = "unknown"
            rcmode = "cq"

            data = calc_stats(infile, ref_path, resolution, qp, fps)
            psnr_dict = data[0]
            ssim_dict = data[1]
            vmaf_dict = data[2]

            df.loc[len(df.index)] = (
                os.path.basename(infile),
                codec,
                resolution,
                width,
                height,
                ref_framerate,
                rcmode,
                quality,
                preset,
                duration,
                actual_bitrate,
                *psnr_dict.values(),
                *ssim_dict.values(),
                *vmaf_dict.values(),
                "-",  # parameters_csv_str,
            )

        # df = df_tmp if df is None else pd.concat([df, df_tmp])
    # write up the results
    df.to_csv(options.outfile, index=False)


tmp_dir = "./"


def calc_stats(enc_path, ref_filename, resolution, qp, fps, debug=0):
    enc_basename = os.path.basename(enc_path)
    # 4. dec: decode copy in order to get statistics
    dec_basename = enc_basename + ".y4m"
    dec_filename = os.path.join(tmp_dir, dec_basename)
    dec_resolution = utils.get_resolution(enc_path)
    ref_resolution = utils.get_resolution(ref_filename)
    ref_pix_fmt = utils.get_pix_fmt(ref_filename)
    # get some stats
    codec = utils.get_codec_name(enc_path)
    encoder_duration = utils.get_duration(enc_path)
    actual_bitrate = utils.get_bitrate(enc_path)

    run_single_dec(enc_path, dec_filename, codec, debug)

    # 5. decs: scale the decoded video to the reference resolution and
    # pixel format
    # This is needed to make sure the quality metrics make sense
    decs_basename = dec_basename + ".scaled"
    decs_basename += ".resolution_%s" % ref_resolution
    decs_basename += ".y4m"
    decs_filename = os.path.join(tmp_dir, decs_basename)
    if debug > 0:
        print("# [%s] scaling file: %s -> %s" % (codec, dec_filename, decs_filename))
    ffmpeg_params = [
        "-y",
        "-nostats",
        "-loglevel",
        "0",
        "-i",
        dec_filename,
        "-pix_fmt",
        ref_pix_fmt,
        "-s",
        ref_resolution,
        decs_filename,
    ]
    retcode, stdout, stderr, _ = utils.ffmpeg_run(ffmpeg_params, debug)
    assert retcode == 0, stderr
    # check produced file matches the requirements
    assert ref_resolution == utils.get_resolution(
        decs_filename
    ), "Error: %s must have resolution: %s (is %s)" % (
        decs_filename,
        ref_resolution,
        utils.get_resolution(decs_filename),
    )
    assert ref_pix_fmt == utils.get_pix_fmt(
        decs_filename
    ), "Error: %s must have pix_fmt: %s (is %s)" % (
        decs_filename,
        ref_pix_fmt,
        utils.get_pix_fmt(decs_filename),
    )

    # get quality scores
    psnr_dict = utils.get_psnr(decs_filename, ref_filename, None, debug)
    ssim_dict = utils.get_ssim(decs_filename, ref_filename, None, debug)
    vmaf_dict = utils.get_vmaf(decs_filename, ref_filename, None, debug)

    # get actual bitrate

    # clean up experiments files
    cleanup = 0
    if cleanup > 0:
        os.remove(dec_filename)
        os.remove(decs_filename)
    # if cleanup > 1:
    #    os.remove(enc_filename)
    return psnr_dict, ssim_dict, vmaf_dict


def run_experiment_single_file(
    infile,
    codecs,
    resolutions,
    rcmodes,
    presets,
    bitrates,
    qualities,
    ref_res,
    ref_pix_fmt,
    gop_length_frames,
    tmp_dir,
    cleanup,
    debug,
):
    # 1. in: get infile information
    if debug > 0:
        print("# [run] parsing file: %s" % (infile))
    assert os.access(infile, os.R_OK), "file %s is not readable" % infile
    in_basename = os.path.basename(infile)
    in_resolution = utils.get_resolution(infile)
    in_framerate = utils.get_framerate(infile)

    # 2. ref: decode the original file into a raw file
    ref_basename = f"{in_basename}.ref_{in_resolution}.y4m"
    if debug > 0:
        print(f"# [run] normalize file: {infile} -> {ref_basename}")
    ref_filename = os.path.join(tmp_dir, ref_basename)
    ref_resolution = in_resolution if ref_res is None else ref_res
    ref_framerate = in_framerate
    ref_pix_fmt = ref_pix_fmt
    ffmpeg_params = [
        "-y",
        "-i",
        infile,
        "-s",
        ref_resolution,
        "-pix_fmt",
        ref_pix_fmt,
        ref_filename,
    ]
    retcode, stdout, stderr, _ = utils.ffmpeg_run(ffmpeg_params, debug)
    assert retcode == 0, stderr
    # check produced file matches the requirements
    assert ref_resolution == utils.get_resolution(
        ref_filename
    ), "Error: %s must have resolution: %s (is %s)" % (
        ref_filename,
        ref_resolution,
        utils.get_resolution(ref_filename),
    )
    assert ref_pix_fmt == utils.get_pix_fmt(
        ref_filename
    ), "Error: %s must have pix_fmt: %s (is %s)" % (
        ref_filename,
        ref_pix_fmt,
        utils.get_pix_fmt(ref_filename),
    )

    columns = (
        "infile",
        "codec",
        "resolution",
        "width",
        "height",
        "framerate",
        "rcmode",
        "quality",
        "preset",
        "encoder_duration",
        "actual_bitrate",
        "psnr_y_mean",
        "psnr_u_mean",
        "psnr_v_mean",
        "psnr_y_p0",
        "psnr_y_p5",
        "psnr_y_p10",
        "psnr_y_p25",
        "psnr_y_p75",
        "psnr_y_p90",
        "psnr_y_p95",
        "psnr_y_p100",
        "psnr_u_p0",
        "psnr_u_p5",
        "psnr_u_p10",
        "psnr_u_p25",
        "psnr_u_p75",
        "psnr_u_p90",
        "psnr_u_p95",
        "psnr_u_p100",
        "psnr_v_p0",
        "psnr_v_p5",
        "psnr_v_p10",
        "psnr_v_p25",
        "psnr_v_p75",
        "psnr_v_p90",
        "psnr_v_p95",
        "psnr_v_p100",
        "ssim_y_mean",
        "ssim_u_mean",
        "ssim_v_mean",
        "ssim_y_p0",
        "ssim_y_p5",
        "ssim_y_p10",
        "ssim_y_p25",
        "ssim_y_p75",
        "ssim_y_p90",
        "ssim_y_p95",
        "ssim_y_p100",
        "ssim_u_p0",
        "ssim_u_p5",
        "ssim_u_p10",
        "ssim_u_p25",
        "ssim_u_p75",
        "ssim_u_p90",
        "ssim_u_p95",
        "ssim_u_p100",
        "ssim_v_p0",
        "ssim_v_p5",
        "ssim_v_p10",
        "ssim_v_p25",
        "ssim_v_p75",
        "ssim_v_p90",
        "ssim_v_p95",
        "ssim_v_p100",
        "vmaf_mean",
        "vmaf_harmonic_mean",
        "vmaf_p0",
        "vmaf_p5",
        "vmaf_p10",
        "vmaf_p25",
        "vmaf_p75",
        "vmaf_p90",
        "vmaf_p95",
        "vmaf_p100",
        "parameters",
    )
    df = pd.DataFrame(columns=columns)

    # run the list of encodings
    for codec, resolution, rcmode, preset in itertools.product(
        codecs, resolutions, rcmodes, presets
    ):
        parameters_csv_str = ""
        for k, v in CODEC_INFO[codec]["parameters"].items():
            parameters_csv_str += "%s=%s;" % (k, str(v))
        # get quality list
        if rcmode == "cbr":
            qualities = bitrates
        elif rcmode == "crf":
            qualities = qualities
        for quality in qualities:
            (
                encoder_duration,
                actual_bitrate,
                psnr_dict,
                ssim_dict,
                vmaf_dict,
            ) = run_single_experiment(
                ref_filename,
                ref_resolution,
                ref_pix_fmt,
                ref_framerate,
                codec,
                resolution,
                quality,
                preset,
                rcmode,
                gop_length_frames,
                tmp_dir,
                debug,
                cleanup,
            )
            width, height = resolution.split("x")
            ref_framerate = utils.get_framerate(ref_filename)
            df.loc[len(df.index)] = (
                in_basename,
                codec,
                resolution,
                width,
                height,
                ref_framerate,
                rcmode,
                quality,
                preset,
                encoder_duration,
                actual_bitrate,
                *psnr_dict.values(),
                *ssim_dict.values(),
                *vmaf_dict.values(),
                parameters_csv_str,
            )
    return df


def run_single_enc(
    infile,
    outfile,
    codec,
    resolution,
    parameter,
    preset,
    rcmode,
    gop_length_frames,
    debug,
):
    if debug > 0:
        print("# [%s] encoding file: %s -> %s" % (codec, infile, outfile))

    # get encoding settings
    enc_tool = "ffmpeg"
    enc_parms = [
        "-y",
    ]
    enc_parms += ["-i", infile]

    enc_env = None
    if CODEC_INFO[codec]["codecname"] == "libsvtav1-raw":
        binary = CODEC_INFO[codec]["binary"]
        quality = parameter
        cmd = f"{binary} --preset {preset} --tbr {quality} --keyint -1 --enable-tpl-la 1 --lp 20 -i {infile} -b {outfile}"
        retcode, stdout, stderr, other = utils.run(cmd, env=enc_env, debug=debug)
        if debug > 0:
            print(stdout)
        assert retcode == 0, stderr
        return other["time_diff"]
    elif CODEC_INFO[codec]["codecname"] == "mjpeg":
        enc_parms += ["-c:v", CODEC_INFO[codec]["codecname"]]
        # TODO(chema): use bitrate as quality value (2-31)
        quality = parameter
        enc_parms += ["-q:v", "%s" % quality]
        enc_parms += ["-s", resolution]
    else:
        enc_parms += ["-c:v", CODEC_INFO[codec]["codecname"]]
        if rcmode == "cbr":
            bitrate = parameter
            # enc_parms += ["-maxrate", "%sk" % bitrate]
            # enc_parms += ["-minrate", "%sk" % bitrate]
            enc_parms += ["-b:v", "%sk" % bitrate]
            # if CODEC_INFO[codec]["codecname"] in ("libx264", "libopenh264", "libx265"):
            #    # set bufsize to 2x the bitrate
            #    bufsize = str(int(bitrate) * 2)
            #    enc_parms += ["-bufsize", bufsize]
        elif rcmode == "crf":
            quality = parameter
            enc_parms += ["-crf", "%s" % quality]

        if CODEC_INFO[codec]["codecname"] in ("libx264", "libx265"):
            # no b-frames
            enc_parms += ["-bf", "0"]
        preset_name = CODEC_INFO[codec].get("preset-name", "preset")
        enc_parms += [f"-{preset_name}", preset]
        enc_parms += ["-s", resolution]
        if gop_length_frames is not None:
            enc_parms += ["-g", str(gop_length_frames)]
        for k, v in CODEC_INFO[codec]["parameters"].items():
            enc_parms += ["-%s" % k, str(v)]
        if CODEC_INFO[codec]["codecname"] in ("libaom-av1",):
            # ABR at https://trac.ffmpeg.org/wiki/Encode/AV1
            enc_parms += ["-strict", "experimental"]

    # pass audio through
    enc_parms += ["-c:a", "copy"]
    enc_parms += [outfile]

    # run encoder
    cmd = [
        enc_tool,
    ] + enc_parms
    retcode, stdout, stderr, other = utils.run(cmd, env=enc_env, debug=debug)
    assert retcode == 0, stderr
    return other["time_diff"]


def run_single_dec(infile, outfile, codec, debug):
    if debug > 0:
        print("# [%s] decoding file: %s -> %s" % (codec, infile, outfile))

    # get decoding settings
    dec_tool = "ffmpeg"
    dec_parms = []
    dec_parms += ["-i", infile]
    dec_env = None
    dec_parms += ["-y", outfile]

    # run decoder
    cmd = [
        dec_tool,
    ] + dec_parms
    retcode, stdout, stderr, _ = utils.run(cmd, env=dec_env, debug=debug)
    assert retcode == 0, stderr


def run_single_experiment(
    ref_filename,
    ref_resolution,
    ref_pix_fmt,
    ref_framerate,
    codec,
    resolution,
    quality,
    preset,
    rcmode,
    gop_length_frames,
    tmp_dir,
    debug,
    cleanup,
):
    if debug > 0:
        print(
            "# [run] run_single_experiment codec: %s resolution: %s "
            "quality: %s rcmode: %s preset: %s"
            % (codec, resolution, quality, rcmode, preset)
        )
    ref_basename = os.path.basename(ref_filename)

    # common info for enc, dec, and decs
    gen_basename = ref_basename + ".ref_%s" % ref_resolution
    gen_basename += ".codec_%s" % codec
    gen_basename += ".resolution_%s" % resolution
    gen_basename += ".quality_%s" % quality
    gen_basename += ".preset_%s" % preset
    gen_basename += ".rcmode_%s" % rcmode

    # 3. enc: encode copy with encoder
    enc_basename = gen_basename + CODEC_INFO[codec]["extension"]
    enc_filename = os.path.join(tmp_dir, enc_basename)
    encoder_duration = run_single_enc(
        ref_filename,
        enc_filename,
        codec,
        resolution,
        quality,
        preset,
        rcmode,
        gop_length_frames,
        debug,
    )

    # 4. dec: decode copy in order to get statistics
    dec_basename = enc_basename + ".y4m"
    dec_filename = os.path.join(tmp_dir, dec_basename)
    run_single_dec(enc_filename, dec_filename, codec, debug)

    # 5. decs: scale the decoded video to the reference resolution and
    # pixel format
    # This is needed to make sure the quality metrics make sense
    decs_basename = dec_basename + ".scaled"
    decs_basename += ".resolution_%s" % ref_resolution
    decs_basename += ".y4m"
    decs_filename = os.path.join(tmp_dir, decs_basename)
    if debug > 0:
        print("# [%s] scaling file: %s -> %s" % (codec, dec_filename, decs_filename))
    ffmpeg_params = [
        "-y",
        "-nostats",
        "-loglevel",
        "0",
        "-i",
        dec_filename,
        "-pix_fmt",
        ref_pix_fmt,
        "-s",
        ref_resolution,
        decs_filename,
    ]
    retcode, stdout, stderr, _ = utils.ffmpeg_run(ffmpeg_params, debug)
    assert retcode == 0, stderr
    # check produced file matches the requirements
    assert ref_resolution == utils.get_resolution(
        decs_filename
    ), "Error: %s must have resolution: %s (is %s)" % (
        decs_filename,
        ref_resolution,
        utils.get_resolution(decs_filename),
    )
    assert ref_pix_fmt == utils.get_pix_fmt(
        decs_filename
    ), "Error: %s must have pix_fmt: %s (is %s)" % (
        decs_filename,
        ref_pix_fmt,
        utils.get_pix_fmt(decs_filename),
    )

    # get quality scores
    psnr_dict = utils.get_psnr(decs_filename, ref_filename, None, debug)
    ssim_dict = utils.get_ssim(decs_filename, ref_filename, None, debug)
    vmaf_dict = utils.get_vmaf(decs_filename, ref_filename, None, debug)

    # get actual bitrate
    actual_bitrate = utils.get_bitrate(enc_filename)

    # clean up experiments files
    if cleanup > 0:
        os.remove(dec_filename)
        os.remove(decs_filename)
    if cleanup > 1:
        os.remove(enc_filename)
    return encoder_duration, actual_bitrate, psnr_dict, ssim_dict, vmaf_dict


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=default_values["debug"],
        help="Increase verbosity (multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    parser.add_argument(
        "--cleanup",
        action="store_const",
        dest="cleanup",
        const=1,
        default=default_values["cleanup"],
        help="Cleanup Raw Files%s"
        % (" [default]" if default_values["cleanup"] == 1 else ""),
    )
    parser.add_argument(
        "--full-cleanup",
        action="store_const",
        dest="cleanup",
        const=2,
        default=default_values["cleanup"],
        help="Cleanup All Files%s"
        % (" [default]" if default_values["cleanup"] == 2 else ""),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_const",
        dest="cleanup",
        const=0,
        help="Do Not Cleanup Files%s"
        % (" [default]" if not default_values["cleanup"] == 0 else ""),
    )
    parser.add_argument(
        "--tmp-dir",
        action="store",
        dest="tmp_dir",
        default=default_values["tmp_dir"],
        help="use TMP_DIR tmp dir",
    )
    parser.add_argument(
        dest="infile_list",
        type=str,
        nargs="+",
        default=default_values["infile_list"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="results file",
    )

    options = parser.parse_args(argv[1:])
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # get infile/outfile
    assert options.outfile != "-"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    run_experiment(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
