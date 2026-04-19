"""
Evaluation script for LeHome manipulation policies.

This script serves as the main entry point for policy evaluation.
Policies are registered via PolicyRegistry and selected with --policy_type.
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import traceback
from lehome.utils.logger import get_logger

from scripts.utils.parser import setup_eval_parser
from scripts.utils.common import launch_app_from_args, close_app

logger = get_logger(__name__)


def main():
    """Main entry point for evaluation script."""
    # 1. AppLauncher can be imported safely here or at top
    from isaaclab.app import AppLauncher

    parser = setup_eval_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    logger.info(f"Starting evaluation for task: {args.task}")
    simulation_app = launch_app_from_args(args)

    try:
        import lehome.tasks
        from scripts.utils.evaluation import eval

        if getattr(args, "headless", False):
            import os
            os.environ["LEHOME_DISABLE_KEYBOARD"] = "1"

        eval(args, simulation_app)
        logger.info("Evaluation finished successfully.")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        traceback.print_exc()
    finally:
        close_app(simulation_app)


if __name__ == "__main__":
    main()
