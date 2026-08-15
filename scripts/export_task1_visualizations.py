"""Export patient-level and site-level Task 1 visualizations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_agent.inference import predict_patient
from resp_agent.localization import localize_suspected_abnormal_segments
from resp_agent.patient_model import load_deployment_model
from resp_agent.visualization import (
    assert_disclaimer_present,
    assert_no_forbidden_wording,
    load_localization_waveform,
    plot_site_evidence_bars,
    plot_site_localization,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Task 1 evidence visualizations for a six-site recording."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=os.environ.get("RESP_AUDIO_DIR"),
        required="RESP_AUDIO_DIR" not in os.environ,
        help="Directory containing the six site recordings.",
    )
    parser.add_argument("--patient-id", default="001")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "task1_visualization_outputs",
    )
    parser.add_argument("--theme", default="paper")
    parser.add_argument("--model", default="site_self_attention")
    return parser.parse_args()


def build_site_audio(audio_dir: Path, patient_id: str) -> dict[str, Path]:
    """Return the expected site-to-file mapping for the VELCRO naming scheme."""
    return {
        f"Site {site_number}": audio_dir / f"00{site_number} ({site_number}).wav"
        for site_number in range(1, 7)
    }


def validate_audio_files(site_audio: dict[str, Path]) -> None:
    missing = [str(path) for path in site_audio.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audio files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    patient_slug = f"patient{args.patient_id}"
    output_dir = args.output_dir.expanduser().resolve() / patient_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    site_paths = build_site_audio(args.audio_dir.expanduser().resolve(), args.patient_id)
    validate_audio_files(site_paths)
    site_audio = {site: str(path) for site, path in site_paths.items()}

    deployment = load_deployment_model(args.model)
    prediction = predict_patient(deployment, site_audio)

    bars_fig = plot_site_evidence_bars(prediction, theme=args.theme)
    assert_no_forbidden_wording(bars_fig)
    assert_disclaimer_present(bars_fig)
    bars_path = save_figure(
        bars_fig,
        output_dir / f"{patient_slug}_site_bars_{args.theme}.png",
    )
    plt.close(bars_fig)
    print(f"Saved {bars_path}")

    for site, audio_path in site_audio.items():
        result = localize_suspected_abnormal_segments(
            deployment,
            site_audio,
            site=site,
        )
        waveform, sample_rate = load_localization_waveform(audio_path)
        fig, metadata = plot_site_localization(
            result,
            waveform,
            sample_rate,
            theme=args.theme,
        )
        assert_no_forbidden_wording(fig)
        assert_disclaimer_present(fig)

        site_number = site.rsplit(" ", 1)[-1]
        output_path = save_figure(
            fig,
            output_dir / f"{patient_slug}_site{site_number}_{args.theme}.png",
        )
        plt.close(fig)
        print(
            f"Saved {output_path} "
            f"(pattern={result['evidence_pattern']}, "
            f"segments={metadata['n_segments_drawn']}, sr={sample_rate})"
        )


if __name__ == "__main__":
    main()
