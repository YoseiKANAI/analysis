subnum = 11
attempt = 5
datafile = "241223"
task = ["NC","FB","D1","D2", "DW"]
weight = [60, 60]
m = {"FB":5, "D1":5, "D2":7, "DW":5}
muscle_columns = ["SO_R", "SO_L", "GM_R", "GM_L", "TA_R", "TA_L", "PL_R", "PL_L", "IO_R", "IO_L", "MF_R", "MF_L"]
muscle_num = len(muscle_columns)
# 利き側 ： 右=>0，左=>1 
domi_arm = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
domi_leg = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]