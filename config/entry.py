import argparse
from config.common import to_shell_defaults
from config.full_ft import TRAINING_PROFILE as FULL_FINETUNE_PROFILE
from config.lora_ft import TRAINING_PROFILE as LORA_FINETUNE_PROFILE
from config.proj_only_ft import TRAINING_PROFILE as PROJECTOR_ONLY_PROFILE


TRAINING_PROFILES = {
    "full_ft": FULL_FINETUNE_PROFILE,
    "lora_ft": LORA_FINETUNE_PROFILE,
    "proj_only_ft": PROJECTOR_ONLY_PROFILE,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=sorted(TRAINING_PROFILES))
    args = parser.parse_args()

    print(to_shell_defaults(TRAINING_PROFILES[args.profile]))


if __name__ == "__main__":
    main()
