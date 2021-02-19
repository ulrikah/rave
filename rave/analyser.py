from rave.template_handler import TemplateHandler
import os
import json
from types import SimpleNamespace


ANALYSER_DIR = "/Users/ulrikah/fag/thesis/rave/rave/feature_extractors"
ANALYSER_BASE = "base_analyser.csd.jinja2"


class Analyser:
    """
    Args:

    feature_extractors: a list of names of which feature extractors to use. Must correspond to the extractors in rave/feature_extractors
    """

    def __init__(
        self,
        feature_extractors: [],
        osc_route=None,
        audio_to_analyse="aOut",
        output_file_path=None,
    ):
        self.feature_extractors = []
        self.global_variables = []
        self.analysis_features = []

        for feature in feature_extractors:
            feature_extractor_template = f"{feature}.csd.jinja2"
            template_path = os.path.join(ANALYSER_DIR, feature_extractor_template)
            meta_info_path = os.path.join(ANALYSER_DIR, f"{feature}.json")
            for path in [template_path, meta_info_path]:
                if not os.path.isfile(path):
                    raise ValueError(f"Couldn't resolve the path to {path}")
            feature_extractor_meta = self._parse_meta_feature_extractor_from_json(
                meta_info_path
            )
            self.analysis_features.extend(feature_extractor_meta.output_channels)
            feature_extractor = TemplateHandler(
                feature_extractor_template, template_dir=ANALYSER_DIR
            ).compile(input=feature_extractor_meta.input)
            self.feature_extractors.append(
                {
                    "name": feature,
                    "csd": feature_extractor,
                    "channels": feature_extractor_meta.output_channels,
                }
            )

            feature_globals_template = f"{feature}.globals.csd.jinja2"
            if os.path.isfile(os.path.join(ANALYSER_DIR, feature_globals_template)):
                self.global_variables.append(
                    TemplateHandler(
                        feature_globals_template, template_dir=ANALYSER_DIR
                    ).compile()
                )

        analyser = TemplateHandler(ANALYSER_BASE, ANALYSER_DIR)
        self.analyser_csd = analyser.compile(
            input=audio_to_analyse,
            feature_extractors=self.feature_extractors,
            global_variables=self.global_variables,
            osc_route=osc_route,
            osc_channels=self.analysis_features,
        )
        if output_file_path is not None:
            analyser.save_to_file(output_file_path)

    def _parse_meta_feature_extractor_from_json(self, feature_extractor_json_path: str):
        try:
            with open(feature_extractor_json_path, "r") as file:
                data = file.read()
                return json.loads(data, object_hook=lambda d: SimpleNamespace(**d))

        except json.decoder.JSONDecodeError as error:
            raise error(
                f"Unable to parse meta information from feature extractor {feature_extractor_json_path}"
            )
