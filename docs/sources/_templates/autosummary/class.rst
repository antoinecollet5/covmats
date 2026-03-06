..
   Template for the html class rendering

   Modified from
   https://github.com/sphinx-doc/sphinx/tree/master/sphinx/ext/autosummary/templates/autosummary/class.rst

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

   {% if attributes %}
   .. rubric:: {{ _('Properties') }}

   .. autosummary::
      :nosignatures:

      {% for item in attributes if not item.startswith('_') %}
      {{ item }}
      {% endfor %}
   {% endif %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:

      {% for item in methods if not item.startswith('_') %}
      {{ item }}
      {% endfor %}
   {% endif %}
