Created a ec2 instance in the default vpc with a public ip
Assigned it the role ec2-disaster-tweets

AWS_EC2_PUBLIC_IP=54.214.109.185

scp -i secrets/aws-disaster-tweets.pem -r ec2_setup ec2-user@${AWS_EC2_PUBLIC_IP}:/home/ec2-user
ssh -i secrets/aws-disaster-tweets.pem ec2-user@${AWS_EC2_PUBLIC_IP}