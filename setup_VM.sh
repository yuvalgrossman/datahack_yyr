# set up Virtual Machine at Google Cloud Platform
#if not cloned already, use:
#git clone https://github.com/yuvalgrossman/datahack_yyr

sudo apt-get update

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
