import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl_ddpg",
        description="Run RL DDPG training.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training and save energy plots and model weights.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference (no training).",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Both modes share the same rollout loop, so at least one is required:
    if not (args.train or args.inference):
        parser.error("one of --train or --inference is required.")

    from rl_ddpg import main as transmon_main

    transmon_main.main(
        train_flag=args.train,
        render_flag=args.render,
        inference_flag=args.inference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
