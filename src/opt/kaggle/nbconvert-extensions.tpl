{#
Jinja template to inject notebook cell metadata to enhance generated HTML output
All cell metadata starting with '_kg_' will be included with its value ({key}-{value})
as a class in the cell's DIV container
#}

{% extends 'index.html.j2' %}
{% block any_cell %}
    <div class="{% for k in cell['metadata'] if k.startswith("_kg_") %}{{k}}-{{cell['metadata'][k] | lower}} {% endfor %}">
        {{ super() }}
    </div>
{% endblock any_cell %}

{% block notebook_css %}
{{ super() }}
<style type="text/css">
div.output_subarea, div.text_cell_render, div.prompt {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
</style>
{% endblock notebook_css%}
