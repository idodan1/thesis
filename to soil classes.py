"""
code that return the class of a soil sample according to usa triangle from:
 https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/?cid=nrcs142p2_054167
"""


def to_soil_class(threesome):
    if threesome['clay'] >= 40:
        if 40 <= threesome['silt'] <= 60:
            return 'silty clay'
        elif 45 <= threesome['sand'] <= 60:
            return 'sandy clay'
        else:
            return 'clay'
    elif 35 <= threesome['clay'] <= 40:
        if 45 <= threesome['sand'] <= 65:
            return 'sandy clay'
        elif 20 <= threesome['sand'] <= 45:
            return 'clay loam'
        else:
            return 'silty clay loam'
    elif 27.5 <= threesome['clay'] <= 35:
        if 45 <= threesome['sand'] <= 72.5:
            return 'sandy clay loam'
        elif 20 <= threesome['sand'] <= 45:
            return 'clay loam'
        else:
            return 'silty clay loam'
    elif 20 <= threesome['clay'] <= 27.5:
        if 52.5 <= threesome['sand'] <= 85:
            return 'sandy clay loam'
        elif 22.5 <= threesome['sand'] <= 52.5 and threesome['silt'] <= 50:
            return 'loam'
        else:
            return 'silt loam'
    elif 15 <= threesome['clay'] <= 20:
        if 52.5 <= threesome['sand']:
            return 'sandy loam'
        elif threesome['silt'] <= 50:
            return 'loam'
        else:
            return 'silt loam'
    elif 10 <= threesome['clay'] <= 15:
        if 70 <= threesome['sand'] <= 85:
            return 'loamy sand'
        if 52.5 <= threesome['sand'] <= 70:
            return 'sandy loam'
        elif 35 <= threesome['silt'] <= 50:
            return 'loam'
        elif 50 <= threesome['silt'] <= 80:
            return 'silt loam'
        else:
            return 'silt'
    elif 0 <= threesome['clay'] <= 10:
        if 85 <= threesome['sand'] <= 100:
            return 'sand'
        elif 70 <= threesome['sand'] <= 85:
            return 'loamy sand'
        elif 42.5 <= threesome['sand'] <= 70 and threesome['silt'] <= 50:
            if 7.5 <= threesome['clay'] <= 10:
                return 'loam'
            else:
                return 'sandy loam'
        elif 42.5 <= threesome['sand'] <= 70 and threesome['silt'] >= 50:
            return 'silt loam'
        elif 50 <= threesome['silt'] <= 80:
            return 'silt loam'
        else:
            return 'silt'


val = {'silt': 50, 'sand': 15, 'clay': 35}
print(to_soil_class(val))
