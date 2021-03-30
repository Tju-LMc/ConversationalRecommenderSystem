echo "train agents"
python3 ./agents/pretrain_AgentRL.py
echo "train belief_tracker"
python3 ./belief_tracker_data/train_belief_tracker.py
echo "train FM"
python3 ./recommendersystem/train_FM.py