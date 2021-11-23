PROJECT_NAME='git@github.com:deanbgarlick/aws-disaster-tweets.git'

eval `ssh-agent -s` # Starts the ssh agent

#Perform a quick update on your instance:
sudo yum update -y

#Install git in your EC2 instance
sudo yum install git -y

python3 -m pip install boto3
python3 ec2_setup/get_secret.py --secret_name github-deanbgarlick-public-ssh-key --output_file ~/.ssh/id_ed25519.pub
python3 ec2_setup/remove_linebreak.py --filename ~/.ssh/id_ed25519.pub
chmod 644 ~/.ssh/id_ed25519.pub
python3 ec2_setup/get_secret.py --secret_name github-deanbgarlick-private-ssh-key --output_file ~/.ssh/id_ed25519
python3 ec2_setup/remove_linebreak.py --filename ~/.ssh/id_ed25519
chmod 400 ~/.ssh/id_ed25519
ssh-add ~/.ssh/id_ed25519
git clone ${PROJECT_NAME}