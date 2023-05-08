"""User interface"""
import datetime

import click

from sademaksit import DATEFMT


@click.command()
@click.option('-t', '--date', help='yyyymmdd')
def cli(date):
    t = datetime.datetime.strptime(date, DATEFMT)
