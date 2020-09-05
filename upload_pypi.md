# 打包python project
```
python3 -m pip install --upgrade pip setuptools wheel
python3 setup.py sdist
```
# 创建配置文件
```
# INSTALL依赖
pip3 install twine
vim ~/.pypirc
===========================
[distutils]
index-servers =
    pypi

[pypi]
repository: https://upload.pypi.org/legacy/
username: <username>
password: <password>
===========================
#上传到
python3 -m twine upload dist/*
```
