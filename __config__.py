import yaml, os, sys

##########################################################
#######  CHECK IF CONFIG FILE EXISTS AND LOAD IT #########
##########################################################
if os.path.exists('config.yml'):
    try:
        with open('config.yml', 'r') as in_file:
            config = yaml.load(in_file, Loader=yaml.Loader)

        input_data_path = config['Data']['input_data_path']
        train_csv_path = config['Data']['train_csv_path']
        val_csv_path = config['Data']['val_csv_path']
        test_csv_path = config['Data']['test_csv_path']
        model_output_path = config['Model']['model_output_path']
        model_name = config['Model']['model_name']
        batch_size = config['Model']['batch_size']
        epochs = config['Model']['epochs']
        target_size = eval(config['Model']['target_size'])
        seed = config['Model']['seed']
        class_mode = config['Model']['class_mode']
        weights = config['Model']['weights']
        input_shape = eval(config['Model']['input_shape'])

    except Exception as e:
        print(e)
        print("Something wrong with the congig file!")
        sys.exit()
else:
    print("Config file not found!")
    sys.exit()

######################################################
##### check if model input and output folders
######################################################
if not os.path.exists(input_data_path):
    os.mkdir(input_data_path)
if not os.path.exists(model_output_path):
    os.mkdir(model_output_path)

######################################################
##### check for input mapping files
######################################################

if not os.path.exists(train_csv_path):
    print("Training csv file does not exist.")
    sys.exit()
if not os.path.exists(val_csv_path):
    print("Validation csv file does not exist.")
    sys.exit()
if not os.path.exists(test_csv_path):
    print("Test csv file does not exist.")
    sys.exit()
