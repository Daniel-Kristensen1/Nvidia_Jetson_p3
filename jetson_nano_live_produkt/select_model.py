from pathlib import Path


def select_model():
    """
    Vælg inpainting-model ud fra nummer:
      1-6  er MobileNetInpainting
      7-9  er UNetInpainting
      10-12 er PConvUNetInpainting

    Returnerer:
      (model_weights_path: Path, model_type: str, fp_bits: int)
      hvor model_type er én af: "mobilenet", "unet", "pconv"
      og fp_bits er 16 eller 32.
    """
    entries = {
        1: ("mobilenet64_med_punish.pth", "mobilenet"),
        2: ("mobilenet64_uden_punish.pth", "mobilenet"),
        3: ("mobilenet128_med_punish.pth", "mobilenet"),
        4: ("mobilenet128_uden_punish.pth", "mobilenet"),
        5: ("mobilenet256_med_punish.pth", "mobilenet"),
        6: ("mobilenet256_uden_punish.pth", "mobilenet"),
        7: ("unet64.pth", "unet"),
        8: ("unet128.pth", "unet"),
        9: ("unet256.pth", "unet"),
        10: ("pconv_unet_lite64_first_trainig_run.pth",  "pconv"),
        11: ("pconv_unet_lite128_first_trainig_run.pth", "pconv"),
        12: ("pconv_unet_lite256_first_trainig_run.pth", "pconv"),
    }

    print("Select a model:")
    for i, (name, mtype) in entries.items():
        print(f"{i}: {name}  [{mtype}]")

    choice = int(input("Enter number (1-12): "))

    filename, model_type = entries[choice]
    modeller_path = Path(__file__).parent / "modeller"
    weights_path = (modeller_path / filename).resolve()

    # ---- Vælg FP16 / FP32 ----
    print("\nSelect precision:")
    print("1: FP32")
    print("2: FP16")
    prec_choice = int(input("Enter number (1-2): "))

    if prec_choice == 1:
        fp_bits = 32
    else:
        fp_bits = 16

    return weights_path, model_type, fp_bits


def find_u2net_model():
    base = Path(__file__).parent
    return (base / "modeller" / "u2net_human_seg.pth").resolve()