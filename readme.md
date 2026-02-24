conda create -n hcm_deguchi_cn python=3.11 -y
conda activate hcm_deguchi_cn
pip install -r requirements.txt


pip install streamlit pandas numpy scikit-learn plotly japanize-matplotlib openpyxl

streamlit run main2.py
streamlit run main0418.py --server.address 0.0.0.0 --server.port 8080
streamlit run main0418_cn.py --server.address 0.0.0.0 --server.port 8080



mkdir -p ~/miniconda3 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh 
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all

sudo yum install screen
screen -S st_session