def load_checkpoints_from_gs(project, source_dir, dst_dir):

    import os 
    import gcsfs
    
    os.makedirs(dst_dir, exist_ok=True)
    fs = gcsfs.GCSFileSystem(project)
    
    for obj in fs.ls(source_dir):
        obj_name = obj.split('/')[-1]
        if obj_name.endswith('.ckpt'):
            print(obj_name)
            fs.get(obj, f'{dst_dir}/{obj_name}')
