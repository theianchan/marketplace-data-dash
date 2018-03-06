
Simple dashboard using Plotly's [Dash](https://dash.plot.ly/) framework.

Data generated from scratch in a notebook [here](https://github.com/theianchan/data-notebooks/blob/master/marketplace-data-generation.ipynb).

CSS for Dash apps need to be hosted externally (which is a real pain). This one is on [CodePen](https://codepen.io/theianchan/pen/yvdbJa.css).

### Deployment notes

Really more for myself. Following instructions at [https://dash.plot.ly/deployment](https://dash.plot.ly/deployment) initially resulted in a rejected push with error `Command "python setup.py egg_info" failed with error code 1 in /tmp/pip-build-8bec89_3/functools32/`.

Solution was to `rm -rf venv`, `python3 -m virtualenv venv`, then `pip3 install dash dash-renderer dash-core-components dash-html-components plotly gunicorn`, updating `requirements.txt` with `pip freeze requirements.txt`.

[Stack Overflow on functools32](https://stackoverflow.com/questions/45168495/deploying-python-flask-app-on-heroku-gives-error-with-functools32). [Stack Overflow on virtualenv in Python 3](https://stackoverflow.com/questions/29934032/virtualenv-python-3-ubuntu-14-04-64-bit). 

### Local development

1. `source venv/bin/activate`
2. In another window, run a server (`http-server -p 3000` or equivalent) to fulfill the externally-hosted stylesheet requirement (like I said, a real pain).
3. Uncomment `# app.css.append_css({"external_url": "http://127.0.0.1:3000/style.css"})`
4. When doen, `deactivate`
