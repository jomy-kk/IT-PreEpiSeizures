import unittest
from datetime import datetime, timedelta
from numpy import array, allclose
from os import remove

from src.biosignals.ECG import ECG
from src.biosignals.HEM import HEM
from src.processing.FrequencyDomainFilter import FrequencyDomainFilter, FrequencyResponse, BandType


class FrequencyDomainFilterTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.samplesx1 = array([440.234375, 356.73828125, 191.69921875, 44.62890625, -74.21875, -126.85546875, -116.30859375, -50.78125, 34.47265625, 119.62890625, 189.74609375, 230.37109375, 275.29296875, 301.66015625, 294.43359375, 277.05078125])
        cls.samplesy1 = array([582.03125, 629.98046875, 620.01953125, 595.8984375, 526.7578125, 402.44140625, 276.5625, 210.15625, 150.390625, 153.125, 167.08984375, 170.41015625, 209.08203125, 244.82421875, 237.01171875, 209.08203125])
        cls.samplesx2 = array([-90.52734375, -92.7734375, -61.62109375, 65.13671875, -8.30078125, 28.22265625, 241.69921875, 187.5, 52.83203125, -58.49609375, 0.390625, 84.86328125, -11.71875, -277.734375, -31.8359375, -15.91796875])
        cls.samplesy2 = array([154.6875, 105.17578125, 60.64453125, 94.82421875, 101.171875, 92.87109375, 119.140625, 127.83203125, 102.34375, 39.74609375, 69.04296875, 128.90625, 103.90625, 74.8046875, 15.91796875, 31.0546875])

        cls.initial1, cls.initial2 = datetime(2018, 12, 11, 11, 59, 5), datetime(2018, 12, 11, 19, 39, 17)  # 1/1/2022 4PM and 3/1/2022 9AM

        cls.sf = 256.0

        cls.n_samplesx = 56736
        cls.n_samplesy = 56736

        cls.channelx, cls.channely = "ecg", "ECG"

        cls.testpath = 'resources/HEM_TRC_tests'
        cls.images_testpath = 'resources/FrequencyFilter_tests'
        cls.biosignal = ECG(cls.testpath, HEM)

    def check_samples(cls, targetx1, targetx2, targety1, targety2):
        # Check sizes
        cls.assertEquals(len(cls.biosignal), 2)
        cls.assertEquals(len(cls.biosignal[cls.channelx][:]), cls.n_samplesx)
        cls.assertEquals(len(cls.biosignal[cls.channely][:]), cls.n_samplesy)
        # Check first 10 samples of each segment
        cls.assertTrue(allclose(cls.biosignal[cls.channelx][cls.initial1:cls.initial1 + timedelta(seconds=16 / cls.sf)].segments[0][:], targetx1))
        cls.assertTrue(allclose(cls.biosignal[cls.channelx][cls.initial2:cls.initial2 + timedelta(seconds=16 / cls.sf)].segments[0][:], targetx2))
        cls.assertTrue(allclose(cls.biosignal[cls.channely][cls.initial1:cls.initial1 + timedelta(seconds=16 / cls.sf)].segments[0][:], targety1))
        cls.assertTrue(allclose(cls.biosignal[cls.channely][cls.initial2:cls.initial2 + timedelta(seconds=16 / cls.sf)].segments[0][:], targety2))

    def check_coefficients(cls, a, b):
        cls.assertTrue(allclose(cls.design.last_denominator_coefficients, a))
        cls.assertTrue(allclose(cls.design.last_numerator_coefficients, b))

    def test_create_filter(cls):
        filter = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, cutoff=35, order=4)

        cls.assertEquals(filter.fresponse, FrequencyResponse.FIR)
        cls.assertEquals(filter.band_type, BandType.LOWPASS)
        cls.assertEquals(filter.cutoff, 35)
        cls.assertEquals(filter.order, 4)

        with cls.assertRaises(AttributeError):
            x = filter.last_numerator_coefficients

        with cls.assertRaises(AttributeError):
            x = filter.last_denominator_coefficients

    def test_undo_filters(cls):
        design = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, cutoff=35, order=4)
        cls.biosignal.filter(design)
        cls.biosignal.undo_filters()
        cls.check_samples(cls.samplesx1, cls.samplesx2, cls.samplesy1, cls.samplesy2)

    def test_apply_lowpass(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, cutoff=35, order=4)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([440.2343749999999, 325.19535959544606, 194.89333667491812, 64.73836768117471, -36.28205628108594, -87.50855522189507, -84.01783558745562, -36.415288228398865, 35.22188242718816, 110.61854433139622, 175.94711792327442, 226.88346297344302, 264.0660408359895, 284.7192184124254, 287.51595784241596, 279.03271635216885])
        filtered_samplesx2 = array([-90.52734374999999, -71.83667554401283, -36.18554850106002, 2.7680297058387096, 34.06492382448229, 84.56978782342406, 139.15108997753347, 133.27847367107123, 68.81408936597155, 15.34194893071686, 8.715746528594849, 1.5624584351862536, -53.22019591979858, -102.28014675208755, -82.83092293961636, -38.73462082560295])
        filtered_samplesy1 = array([582.0312499999999, 604.1328990675064, 603.2359926934516, 569.2891979485448, 499.23590570891145, 401.78557442078784, 302.3707093641662, 225.1011270110008, 179.2593375070178, 163.52553233573627, 167.84084687089984, 183.72148237788448, 206.68285075721707, 224.54142256936456, 224.99021208662447, 211.4497900103218])
        filtered_samplesy2 = array([154.68749999999997, 114.63966697082057, 91.6375241183674, 89.78723931349278, 96.30968735085729, 103.66624229041027, 111.15609195703396, 109.38806738497382, 92.42020731269847, 76.7174665151307, 82.20272530841643, 96.23923951986738, 92.10531555748085, 68.40893058182145, 46.422420730036954, 44.112122441788145])
        a = array([1])
        b = array([0.02253315, 0.23286143, 0.48921084, 0.23286143, 0.02253315])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()

    def test_apply_highpass(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.HIGHPASS, cutoff=35, order=4)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([92.60275277843985, 91.28829000917298, 36.818519425204734, 4.406778831424174, -27.529196890329423, -37.91913093415782, -34.285788382527436, -13.950675057813184, 7.409505153476878, 27.978214612829056, 45.451705623223035, 47.14528699203644, 61.14625493387997, 69.79861205288151, 63.63123341283505, 55.94450292247304])
        filtered_samplesx2 = array([-19.042314069568274, -26.266114826445353, -31.944425786264997, 54.28775344788265, -23.74202581025071, -30.179160991832646, 103.49514009937701, 54.81175531849776, 6.226774228110688, -45.876405049343, -5.405675794620724, 54.45303706785619, 31.423674662817465, -160.2457921064001, 40.10140395525448, 2.344019001831333])
        filtered_samplesy1 = array([122.42954892624263, 143.62423199570514, 132.3112330567505, 133.37015522086315, 121.31035598748181, 84.83938146380942, 46.77511466137257, 43.701548904740655, 19.903420101829802, 30.537849206377977, 37.70801408037718, 28.835884820779274, 44.37732039678088, 59.90635960322358, 53.587060078912536, 42.79563088923774])
        filtered_samplesy2 = array([32.538323070330236, 21.5035619995318, -2.593721116339816, 25.84969079187906, 24.376382027783066, 12.446148897612465, 28.258439633209264, 33.630707594824315, 29.070239433399195, -9.843251814548585, 8.508523612895296, 43.868406875890145, 22.370677230954552, 21.67615362333569, -11.185350757739387, 0.9788250778933211])
        a = array([1])
        b = array([-0.01309641, -0.13534052,  0.75551178, -0.13534052, -0.01309641])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()

    def test_apply_bandpass(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, cutoff=(5,35), order=4)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([511.7986340153492, 378.56174491626234, 226.6092506308287, 74.48431264081334, -43.506148052528786, -103.18571394811204, -98.83712018964874, -42.951301490785596, 40.87048143292452, 128.88934017436787, 204.93544110189444, 264.10231179121786, 307.4203403051477, 331.491022039846, 334.54451151569964, 324.4791062304942])
        filtered_samplesx2 = array([-105.24341919525924, -84.06504460446884, -42.15362522760199, 3.5854950914590407, 38.95463404981099, 98.0083929438049, 163.57340904539572, 156.68286035680282, 79.60037436591982, 16.33878199044886, 10.13294380332879, 3.4151436203239323, -61.97159826391307, -121.09206361916416, -96.87670844976184, -44.4115353002712])
        filtered_samplesy1 = array([676.6459313956257, 702.9784300944913, 702.1596928966627, 662.7927573413829, 581.1543692000474, 467.14372601310504, 350.90384078116153, 260.85998199606547, 207.59579258515396, 189.6011244087791, 194.84737625961122, 213.37333936178763, 240.408096753616, 261.5447396163609, 261.95848562393604, 245.80016131486673])
        filtered_samplesy2 = array([179.83341532393814, 132.79928306227907, 105.9280241605399, 104.19390999892795, 111.97240636047869, 120.47593466274178, 129.4699574372695, 127.64088362719183, 107.43143785301363, 88.52514727589006, 95.29672987304015, 112.4436237235112, 107.65164174111443, 79.4776201909135, 53.36823870236009, 50.970819159570205])
        a = array([1])
        b = array([0.02172166, 0.24946488, 0.53584742, 0.24946488, 0.02172166])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()

    def test_apply_bandstop(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDSTOP, cutoff=(5,35), order=4)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([440.2343749999999, 405.5671205035317, 181.405484527368, 28.485252571327493, -111.89163863526858, -162.73995074193448, -147.4814237795768, -61.59850870371859, 34.78729243383115, 128.46467654148972, 206.81555310328048, 227.15023050461008, 285.66483385690213, 321.28334470809784, 299.9977097195605, 270.34613270818625])
        filtered_samplesx2 = array([-90.52734374999996, -113.62991253108876, -117.80596227137889, 185.44019548376173, -74.32867083240272, -78.7641679918031, 400.3313491654663, 235.97596325840985, 37.7616304002299, -160.2433887260206, -15.72053408460598, 195.55029865206595, 88.12920796755944, -582.0798898559642, 105.13127545509255, 1.8478735063006475])
        filtered_samplesy1 = array([582.0312499999999, 663.9369951274633, 627.2407701169179, 621.3940480684403, 559.1633131304958, 403.09071160502566, 241.77105800424982, 207.09474871949453, 114.26232039807664, 147.22694203458624, 174.09640114987465, 149.3481460628589, 210.5048100843466, 270.55081669123234, 248.75287275858187, 205.5583328299093])
        filtered_samplesy2 = array([154.6874999999999, 102.46083586711056, 14.238221348942297, 111.86448573685777, 110.28792078066269, 71.90867567857346, 129.0529745410709, 148.551722322824, 124.5584467653288, -15.146069607644154, 50.81163037563672, 179.34760690488636, 106.46560072254276, 92.25027651613344, -28.054034102046373, 14.095950563316283])
        a = array([1])
        b = array([-0.01798059, -0.20650009,  1.44896137, -0.20650009, -0.01798059])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()


    def test_apply_butterworth(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.BUTTER, BandType.LOWPASS, cutoff=35, order=4)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([440.21905152275974, 304.4795479867062, 169.94073917123427, 45.27776717920752, -52.75059616934851, -106.43780944783379, -106.7537391416006, -58.75266276416283, 20.365547901855702, 108.61305363037368, 187.22498086689149, 245.2991858088813, 280.0331208906721, 294.19980084493506, 293.15353765306344, 282.63632987786224])
        filtered_samplesx2 = array([-90.42799215125021, -72.44545332247914, -44.05716881847533, -0.7927782866013686, 52.26580820944758, 101.46126296462884, 130.49371234124786, 129.06533575718052, 98.56741470016361, 50.21733503294118, -1.9315067313839158, -46.04820630781686, -73.19545560674891, -78.19093822303068, -63.569792054971686, -41.65987147160983])
        filtered_samplesy1 = array([582.0162429579536, 615.4242461432657, 622.8654412832713, 588.1830013220928, 510.98103728058777, 406.30315284775287, 298.27215899050003, 210.57100225664084, 158.07479968334746, 143.05708837786636, 156.71466341112696, 184.2404494049144, 210.8172592471691, 226.4545687014094, 228.33129886665654, 219.90561910276358])
        filtered_samplesy2 = array([154.79201385408, 117.38223027929138, 91.91998673758638, 83.87228426503424, 89.82081897027595, 100.56800038829138, 107.17319687329767, 105.92084959172047, 99.22341636592581, 92.16908049256456, 87.63468372090394, 84.22692702520143, 78.71258627863203, 69.99357949109411, 60.2088482755857, 52.10103777289346])
        a = array([1., -1.78148382,  1.50594155, -0.60229749,  0.09733974])
        b = array([0.01371875, 0.054875  , 0.08231249, 0.054875  , 0.01371875])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()

    def test_apply_chebyshev1(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.CHEBY1, BandType.LOWPASS, rp=1, cutoff=35, order=4)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([349.93495118557433, 227.3033996299854, 103.71552384380104, -13.647009774296144, -108.3102208684852, -160.96327768161365, -159.72318302777919, -107.51866738076204, -21.926867574593594, 72.26593344930406, 153.18440715884267, 209.06281981119693, 239.40838470639747, 250.9064871599279, 251.51956915848908, 246.3436133547194])
        filtered_samplesx2 = array([-64.43258311538743, -49.56999415591678, -28.778508687123075, 5.569728462879141, 52.68015174606167, 102.06337105641155, 137.20354917549503, 142.90606701931281, 113.00714817399182, 54.620172884456345, -13.42586868530874, -67.77998168281401, -90.83499825495694, -79.65379024343886, -48.39038979359826, -21.01768685197116])
        filtered_samplesy1 = array([462.0242141651657, 497.54719950932633, 508.169586462266, 478.61609436146745, 409.24864024756715, 315.28179767603893, 220.0172330895614, 145.36343640346672, 103.92131352169883, 95.83358276274261, 111.08581000659515, 135.40610301544464, 156.59789007501655, 168.49081936579256, 171.2603372145859, 168.75811105384136])
        filtered_samplesy2 = array([124.31569493505965, 87.87045664876801, 63.48000717511291, 57.086903907863395, 65.20245672784186, 78.12905739610487, 86.41988821852908, 86.09952379605026, 79.49229728090957, 71.85889449940156, 66.91310911325826, 64.62355435348482, 62.418726797499815, 58.04498011835593, 51.33713473441822, 43.68981893796707])
        a = array([ 1.        , -2.56534373,  2.99338263, -1.76518687,  0.44350943])
        b = array([0.00592467, 0.02369869, 0.03554803, 0.02369869, 0.00592467])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()

    def test_apply_chebyshev2(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.CHEBY2, BandType.LOWPASS, rs=1, cutoff=35, order=4)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([440.033405206162, 330.2782769599933, 181.78420331499532, 48.11275963943034, -67.12559704761684, -116.90102876186685, -108.07184995943075, -53.85247398353445, 22.66642024993719, 108.10894835495458, 189.16446034688596, 244.25777408178723, 284.69210245831766, 299.3089828984628, 290.6697900819345, 272.60630850574995])
        filtered_samplesx2 = array([-72.38683782798307, -34.89230581185972, -62.575858591647744, -12.426432353508718, -6.814039621832812, 80.13054119711084, 220.01961141521105, 165.5892138635972, 87.36665365264099, -5.984071965532241, -31.965118118067025, -16.114962198054656, -12.08671890070342, -162.83770952708312, -40.285634951406045, -40.11966407843957])
        filtered_samplesy1 = array([582.0564850253243, 622.4223403085098, 625.6702809192254, 598.4718998038421, 519.620312705144, 403.19618216646876, 281.7788705868254, 207.15665769503522, 153.17151680192876, 151.2657923044551, 162.06909559326192, 176.32361271827781, 212.3285184716096, 238.25785055699959, 234.4112061760328, 213.35592346816708])
        filtered_samplesy2 = array([156.3310774500165, 119.76998352303883, 84.03399137889829, 90.82720514031989, 83.41287842825395, 88.71309723386807, 115.90870005573431, 118.94904573359156, 109.21553366691498, 71.3633206128373, 75.98894256918354, 101.61565016134324, 84.33165157171554, 71.42242867151205, 34.24044316056747, 47.49528640621516])
        a = array([ 1.        , -0.97153399,  1.247486  , -0.47387806,  0.52002154])
        b = array([ 0.71116956, -0.60816826,  1.11609289, -0.60816826,  0.71116956])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()

    def test_apply_elliptic(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.ELLIP, BandType.LOWPASS, cutoff=35, order=4, rp=1, rs=1)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([6942435.303953765, 11839058.188819114, 14257957.072291976, 13702663.715635203, 10308195.409267912, 4799936.503223982, -1671601.1676694218, -7776036.722063037, -12272915.60042842, -14253757.047427826, -13315781.794110822, -9640653.368733859, -3964936.0758584053, 2556860.061522986, 8578369.858478794, 12838133.028961767])
        filtered_samplesx2 = array([21272130.828110605, 24881232.309730154, 23380242.63134039, 16940863.159208547, 6822985.072412525, -4838830.355332321, -15504071.88118165, -22835498.999718793, -25268196.01423836, -22364874.038976733, -14863296.796856437, -4434111.037659023, 6742358.177103768, 16442129.280156318, 22800205.596144527, 24604017.551508356])
        filtered_samplesy1 = array([-9510757.954663228, -4955429.373884554, 591552.1230893009, 5978635.897702336, 10111877.13361669, 12167915.478560438, 11743660.854218284, 8925187.556568995, 4271038.571695342, -1287333.8652630385, -6617830.407856988, -10610361.649544109, -12411362.371964367, -11618457.430069199, -8383188.803786827, -3387823.645775818])
        filtered_samplesy2 = array([-7459683.3235978, 23585258.288870018, 49504837.16320816, 64780383.49887003, 66358605.54920198, 54213634.728014976, 31156131.802110314, 2057059.107075471, -27226120.010738216, -50955460.3441071, -64490939.57436721, -65074986.55431929, -52375279.256899014, -28691562.817533843, 1306433.1432720982, 31395229.608532015])
        a = array([ 1.        , -3.14030613,  4.41565464, -3.14030613,  1.        ])
        b = array([ 0.06642097, -0.07162779,  0.13077028, -0.07162779,  0.06642097])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()

    def test_apply_bessel(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.BESSEL, BandType.LOWPASS, cutoff=35, order=4)
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([440.23388775112886, 315.68203960936205, 197.51020766840128, 94.29235015877767, 16.301442639341097, -27.81164289494302, -34.84487260031746, -8.796542921963649, 40.41099005819904, 100.27974256787292, 159.2864356869191, 209.14449238272667, 245.41000642206131, 266.99740097036965, 275.3139523693925, 273.31991074579486])
        filtered_samplesx2 = array([-90.52811430020633, -60.75711428638602, -28.348289026480344, 7.191077499320952, 42.92944943735487, 72.66066590708502, 89.08082908331613, 87.69400775072042, 69.77235294626135, 41.41057970766394, 9.678919464902062, -19.86974447692035, -42.7127071703568, -54.94253229594646, -55.28952273041838, -47.08135406603423])
        filtered_samplesy1 = array([582.0316799649624, 579.7293758178268, 564.6908800931637, 528.7735371965191, 471.3273681400764, 399.3183941788144, 324.6235679435826, 259.63942775663037, 213.14886746259089, 188.12066796723178, 181.90154801294565, 188.0902741802157, 199.04256517475915, 208.2542941283371, 212.06257891244752, 210.04988442551556])
        filtered_samplesy2 = array([154.68611853702708, 130.93522615688516, 112.54367326502378, 102.23099190940542, 99.10794901536711, 99.75430461448204, 100.40481591884355, 98.91182933580053, 95.4206239485782, 91.35133666571312, 87.51036555192859, 83.15262353381196, 76.99391981199275, 68.94426015576121, 60.447204413865414, 53.06247672333541])
        a = array([ 1.        , -1.63898213,  1.19375273, -0.42588832,  0.0612522 ])
        b = array([0.01188341, 0.04753362, 0.07130043, 0.04753362, 0.01188341])

        cls.check_coefficients(a, b)
        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()

    def test_plot_bode(cls):
        cls.design = FrequencyDomainFilter(FrequencyResponse.BESSEL, BandType.LOWPASS, cutoff=35, order=4)
        cls.biosignal.filter(cls.design)
        test_image_path = cls.images_testpath + "/testplot.png"
        cls.design.plot_bode(show=False, save_to=test_image_path)

        #with open(cls.images_testpath + "/bode_bessel_lowpass_25Hz_4th_256Hz.png", 'rb') as target, open(test_image_path, 'rb') as test:
        #    cls.assertEquals(target.read(), test.read())

        remove(test_image_path)


if __name__ == '__main__':
    unittest.main()
