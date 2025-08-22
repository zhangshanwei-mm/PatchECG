for i in range(6):
    print(i)
    
        val_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(val_delete_data))
    test_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(test_delete_data))
    
    temp_val_data = list()
    for i in range(len(val_delete_data)):
        temp_layout_data = gen_true_ecg_layout(val_delete_data[i],length = 1000,layout = val_list[i])
        temp_val_data.append(temp_layout_data)
    
    temp_test_data = list()
    for i in range(len(test_delete_data)):
        temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = test_list[i])
        temp_test_data.append(temp_layout_data)