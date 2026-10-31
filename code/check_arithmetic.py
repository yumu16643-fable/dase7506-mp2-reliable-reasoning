"""Behavioral checks for numerical parsing, rounding, units and code rejection."""
from arithmetic_features import equations,number_value

if __name__=='__main__':
    tests=[('3 * 20 = 60',False),('3 * 20 = 100',True),('0.6 / 9 = 2 / 15',True),('0.6 / 9 = 1 / 15',False),(r'\\frac{0.6}{9} = \\frac{2}{15}',True),(r'\\frac{0.6}{9} = \\frac{1}{15}',False),('1 / 7 = 0.14',False),('1,200 * 2 = 2,400',False),('20% * 50 = 10',False),('2 ^ 3 = 8',False),('94 - 46 = 48',False),('(-3) * 2 = -6',False)]
    for text,bad in tests:
        text=text.replace('\\\\','\\'); records=equations(text); assert records,(text,'No parsed checks');assert any(r['bad'] for r in records)==bad,(text,records)
    for text in ['0.6 hours = 36 minutes',r'0.6 \\text{ hours} = 36 \\text{ minutes}','x2 = 4','x = 3','2 + x = 4','2 + x + 3 = 5','x - 3 = 5','3 * 2 = 2x','2 + variable - 3 = 5']:
        assert not equations(text.replace('\\\\','\\')),(text,equations(text))
    for expression in ["__import__('os').system('x')",'2**999999','1/0','[1,2]','True','2//1']:
        try:number_value(expression)
        except Exception:pass
        else:raise AssertionError(expression)
    print('Arithmetic, LaTex, rounding, units, identifier rejection and bounded-expression checks passed.')
