from jinja2 import Environment, FileSystemLoader


class TemplateHandler:
    def __init__(self, template, template_dir):
        self.env = Environment(autoescape=False, loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template(template)

    def compile(self, *args, **kwargs):
        self.csd = self.template.render(*args, **kwargs)
        return self.csd

    def save_to_file(self, output_file_path):
        with open(output_file_path, "w") as output_file:
            output_file.write(self.csd)
