import click
from hyperlpr3.command.aliased_group import AliasedGroup
from hyperlpr3.command.sample import sample

__all__ = ['cli']

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(cls=AliasedGroup, context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(sample)

if __name__ == '__main__':
    cli()