"""User interface"""
import datetime

import click

from sademaksit import DATEFMT, ls_low_elev, maxit


@click.command()
@click.argument('yyyymmdd')
@click.option('-i', '--input-dir', metavar='DIR', help='directory containing hdf5 input data')
@click.option('-o', '--output-dir', metavar='DIR', help='write output here')
@click.option('-s', '--size', metavar='PX', help='output raster size PX*PX', default=1024)
@click.option('-r', '--resolution', metavar='METRE', help='spatial resolution in meters')
def cli(yyyymmdd, input_dir, output_dir, size, resolution):
    if resolution is None:
        if size>1999:
            resolution = 250
        elif size>999:
            resolution = 500
        elif size>499:
            resolution = 1000
        else:
            resolution = 2000
    date = datetime.datetime.strptime(yyyymmdd, DATEFMT)
    h5paths = ls_low_elev(date, input_dir)
    maxit(date, h5paths, output_dir, size=size, resolution=resolution)
