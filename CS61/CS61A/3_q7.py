def repeating ( t , n ) :                       # 检测数字n是否是长度为t的数字count_t的重复结构

    power_t = 10 ** t                           # 为后面的提取数字做准备

    count_n , L = n , 0                         # count_n要被计算,n不能动
    while count_n > 0 :                         # 计算n有多少位数字,因为只有当n的位数是t的整数倍的时候才有可能True
        count_n = count_n // 10                 # 给count_n减一位
        L = L + 1                               # 给长度加一
    
    if L % t != 0 :                             # 判断L是否是t的整数倍
        return False                            # 若不是整数倍,直接返回False
    
    count_t = n % power_t                       # 提取n的后t位数字  这个地方ai说 count_t = n % 10 ** (L -t )比较好,也就是取最高位的比较好,这是为何

    if power_t > count_t > power_t // 10 :      # 保证不出现以0为首的重复数字

        temp_n = 0 
        while n > 0 :                           # 开始查看每一组数字都和重复数组一样 
            temp_n = n % power_t                #temp_n等于n的最后t位数字

            if temp_n != count_t :              # 如果n的最后t位不等于这个数组就返回False
                return False                    # 不等于就False
            n = n // power_t                    # 去除n的最后t位

        return True                             # 没问题就True
    
    else : 
        return False                            # 首位为0,直接False