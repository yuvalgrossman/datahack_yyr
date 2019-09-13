# set up Virtual Machine at Google Cloud Platform
#if not cloned already, use:
#git clone https://github.com/yuvalgrossman/datahack_yyr

sudo apt-get update
# install python 3.7: 
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
sudo tar xzf Python-3.7.4.tgz
cd Python-3.7.4
sudo ./configure --enable-optimizations
sudo make altinstall
python3.7 -V

cd ../../../home/grosman/
# install pip: 
sudo python3.7 -m pip install pip
sudo python3.7 -m pip install --upgrade pip  

#install virtualenv:
sudo apt install virtualenv

#create virtual environment:
cd datahack_yyr
virtualenv -p python3.7 venv
source venv/bin/activate

#install requirements
pip install -r requirements.txt

#configure jupyter: 
#jupyter notebook --generate-config
#vi ~/.jupyter/jupyter_notebook_config.py
# add these lines to the file, save and exit:
#c = get_config()
#c.NotebookApp.ip = '*'
#c.NotebookApp.open_browser = False
#c.NotebookApp.port = 7000

#launch jupyter: 
#jupyter-notebook --no-browser --port=7000 &
