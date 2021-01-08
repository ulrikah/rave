from jinja2 import Environment, Template, FileSystemLoader


class TemplateHandler:
    def __init__(self, template, template_dir="rave/effects"):
        # get the jinja going
        self.env = Environment(
            autoescape=False, loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template(template)

    def compile(self, *args, **kwargs):
        self.csd = self.template.render(*args, **kwargs)
        return self.csd
