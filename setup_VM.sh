# set up Virtual Machine at Google Cloud Platform
#if not cloned already, use:
#git clone https://github.com/yuvalgrossman/datahack_yyr

# install python 3.7: 
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7

# install pip: 
sudo apt-get install python3-pip
pip3 install --upgrade pip  

#install virtualenv:
sudo apt install virtualenv

#create virtual environment:
cd datahack_yyr
virtualenv -p python3 venv
source venv/bin/activate

#install requirements
pip install -r requirements.txt

# install modin from github, due to a problem with current release: 
pip uninstall modin
pip install git+https://github.com/modin-project/modin

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
