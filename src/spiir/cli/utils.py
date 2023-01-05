import functools
import logging

import click


def click_logger_options(f):
    """A decorator function to include click options for logging arguments."""
    f = click.option("--log-level", "log_level", type=int, default=logging.WARNING)(f)
    f = click.option("--verbose", "log_level", type=int, flag_value=logging.INFO)(f)
    f = click.option("--debug", "log_level", type=int, flag_value=logging.DEBUG)(f)
    f = click.option("--log-file", type=click.Path(writable=True), default=None)(f)

    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options


class DefaultCommandGroup(click.Group):
    """Enable a default command for a group.

    Allows the user to specify a default command to for the CLI without having
    to specify an argument. Source: https://stackoverflow.com/a/52069546.
    """

    def command(self, *args, **kwargs):
        default_command = kwargs.pop("default_command", False)
        if default_command and not args:
            kwargs["name"] = kwargs.get("name", "<>")
        decorator = super(DefaultCommandGroup, self).command(*args, **kwargs)

        if default_command:

            def new_decorator(f):
                cmd = decorator(f)
                self.default_command = cmd.name
                return cmd

            return new_decorator

        return decorator

    def resolve_command(self, ctx, args):
        try:
            # test if the command parses
            return super(DefaultCommandGroup, self).resolve_command(ctx, args)
        except click.UsageError:
            # command did not parse, assume it is the default command
            args.insert(0, self.default_command)
            return super(DefaultCommandGroup, self).resolve_command(ctx, args)
