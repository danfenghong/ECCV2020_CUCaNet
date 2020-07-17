import importlib
from model.base_model import BaseModel


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "model." + model_name
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def create_model(opt, hsi_c, msi_c, lrhsi_h, lrhsi_w, sp_matrix, sp_range):
    model_class = find_model_using_name(opt.model_name)
    instance = model_class()
    instance.initialize(opt, hsi_c, msi_c, lrhsi_h, lrhsi_w, sp_matrix, sp_range)
    return instance
