function myframe=donghuatest2(a,sita)
    Myfiguredonghua;
    hold on;
    %myblue=[36/255 90/255 165/255];
    R=36+(a-0.5)/0.5*(235-36);
    G=90+(a-0.5)/0.5*(92-90);
    B=165+(a-0.5)/0.5*(61-165);
    myblue=[R/255 G/255 B/255];
    %myblue=[235/255 92/255 61/255];
    %开始画日子旁
    z=1181;
    %[B_rest1_x,B_rest1_y]=myinv_fill_donghua([539,z-487;529,z-456;522,z-433;
    %    515,z-422;475,z-400;450,z-392;414,z-395;362,z-412;329,z-428;317,z-452;310,z-478;
    %    307,z-504;307,z-530;307,z-545;307,z-565;320,z-618;342,z-650;368,z-668;369,z-674;
    %    392,z-672;399,z-674;428,z-674;454,z-672;487,z-668;513,z-658;
    %    529,z-645;536,z-636;543,z-610;547,z-558;539,z-487;539,z-487]); %日的外围,最后一项决定是否包裹
    [B_rest1_x,B_rest1_y]=myinv_fill_donghua2(a,sita,[539,z-487;529,z-456;
        515,z-422;475,z-400;450,z-392;414,z-395;362,z-412;329,z-428;310,z-478;
        307,z-530;320,z-618;342,z-650;368,z-668;369,z-674;
        392,z-672;428,z-674;454,z-672;487,z-668;513,z-658;
        529,z-645;543,z-610;547,z-560;539,z-487;]); %日的外围,最后一项决定是否包裹
    patch00=patch(B_rest1_x,B_rest1_y,myblue);
    set(patch00,{'LineStyle'},{'none'}); %设置颜色和线

    %日的上圈
    [B_rest2_x,B_rest2_y]=myinv_fill_donghua2(a,sita,[355,z-473;359,z-467;362,z-463;383,z-452;401,z-447;430,z-444;460,z-445;
        481,z-460;488,z-481;488,z-499;483,z-506;473,z-509;
        464,z-512;443,z-514;432,z-514;402,z-516;372,z-518;355,z-519;
        351,z-513;350,z-502;350,z-487;355,z-473]);

    %日的下圈
    % [B_rest3_x,B_rest3_y]=myinv_fill_donghua([383,z-620;402,z-621;422,z-621;440,z-621;451,z-621;
    %     460,z-619;471,z-615;478,z-611;480,z-600;492,z-581;493,z-568;
    %     494,z-558;487,z-552;473,z-552;441,z-554;410,z-557;388,z-560;
    %     366,z-564;360,z-570;357,z-576;356,z-582;357,z-587;
    %     362,z-600;371,z-610;382,z-617;386,z-620;383,z-620])
    [B_rest3_x,B_rest3_y]=myinv_fill_donghua2(a,sita,[383,z-620;402,z-621;422,z-621;440,z-621;
        460,z-619;480,z-600;493,z-568;
        487,z-552;441,z-554;410,z-557;388,z-560;
        366,z-564;357,z-576;357,z-587;
        362,z-600;382,z-617;383,z-620]);
    patch0=patch( [B_rest3_x B_rest2_x],[ B_rest3_y B_rest2_y ],[1 1 1]);
    set(patch0,{'LineStyle'},{'none'}); %设置颜色和线


    %日下一横
    [B_rxyh_x,B_rxyh_y]=myinv_fill_donghua2(a,sita,[574,z-701;480,z-703;434,z-707;368,z-714;332,z-725;318,z-740;
        317,z-746;321,z-754;339,z-760;350,z-762;387,z-762;448,z-756;473,z-754;
        527,z-747;566,z-746]);
    patch9=patch(B_rxyh_x,B_rxyh_y,myblue);
    set(patch9,{'LineStyle'},{'none'}); %设置颜色和线

    %横下竖提上去
    [B_hxst_x,B_hxst_y]=myinv_fill_donghua2(a,sita,[566,z-746;581,z-800;590,z-853;599,z-896;624,z-928;
        640,z-931;651,z-916;648,z-888;638,z-837;635,z-817;631,z-775;
        630,z-724;633,z-701;641,z-638;650,z-590;660,z-537;670,z-478;
        671,z-459;661,z-425;657,z-391;655,z-374;655,z-374;655,z-374;647,z-327;647,z-327;647,z-327;610,z-378;614,z-422;
        587,z-450;565,z-469;539,z-486;539,z-486;
        539,z-486;547,z-560;588,z-516;612,z-494;
        620,z-488;604,z-556;589,z-637;574,z-701;566,z-746]);
    patch8=patch(B_hxst_x,B_hxst_y,myblue);
    set(patch8,{'LineStyle'},{'none'}); %设置颜色和线

    %左上角那一撇
    [B_zsjp_x,B_zsjp_y]=myinv_fill_donghua2(a,sita,[610,z-378;592,z-391;573,z-403;561,z-409;546,z-406;
        538,z-399;537,z-391;552,z-375;571,z-353;596,z-314;
        613,z-282;627,z-257;638,z-241;658,z-237;667,z-243;
        671,z-261;668,z-274;658,z-299;647,z-327]);
    patch7=patch(B_zsjp_x,B_zsjp_y,myblue);
    set(patch7,{'LineStyle'},{'none'}); %设置颜色和线

    %最上边那一横
    [B_zsbh_x,B_zsbh_y]=myinv_fill_donghua2(a,sita,[647,z-327;704,z-328;752,z-324;786,z-323;819,z-323;859,z-323;878,z-323;
        895,z-326;908,z-336;897,z-358;853,z-362;815,z-362;735,z-364;701,z-367;
        671,z-372;655,z-371]);
    patch6=patch(B_zsbh_x,B_zsbh_y,myblue);
    set(patch6,{'LineStyle'},{'none'}); %设置颜色和线

    %右上圈外圈

    [B_ysw_x,B_ysw_y]=myinv_fill_donghua2(a,sita,[737,z-386;774,z-383;794,z-383;843,z-384;880,z-386;912,z-403;
        921,z-423;917,z-454;892,z-479;854,z-485;806,z-486;766,z-485;
        727,z-481;704,z-470;689,z-452;685,z-424;710,z-391;737,z-386]);
    %右上圈内圈
    [B_ysn_x,B_ysn_y]=myinv_fill_donghua2(a,sita,[738,z-419;776,z-418;813,z-418;845,z-419;872,z-423;881,z-433;
        882,z-440;875,z-447;860,z-451;841,z-452;822,z-453;781,z-452;744,z-451;
        733,z-446;727,z-438;727,z-428;734,z-421;738,z-419]);
    patch4=patch([B_ysw_x B_ysn_x],[B_ysw_y B_ysn_y],myblue);
    set(patch4,{'LineStyle'},{'none'}); %设置颜色和线

    %右中圈外圈
    [B_yzw_x,B_yzw_y]=myinv_fill_donghua2(a,sita,[726,z-514;769,z-512;816,z-510;864,z-512;896,z-518;
        920,z-545;921,z-580;893,z-607;861,z-613;804,z-614;761,z-613;
        719,z-609;686,z-576;694,z-530;726,z-514]);
    %右中圈内圈
    [B_yzn_x,B_yzn_y]=myinv_fill_donghua2(a,sita,[729,z-548;763,z-542;796,z-542;844,z-543;874,z-549;
        882,z-558;871,z-576;837,z-581;806,z-582;734,z-576;
        727,z-551;736,z-546;729,z-548]);
    patch3=patch([B_yzw_x B_yzn_x],[B_yzw_y B_yzn_y],myblue);
    set(patch3,{'LineStyle'},{'none'}); %设置颜色和线
    %右下角外圈,从上边的竖开始,
    [B_yxw_x,B_yxw_y]=myinv_fill_donghua2(a,sita,[836,z-632;838,z-663;846,z-673;
        859,z-671;878,z-662;887,z-651;892,z-634;909,z-625;926,z-635;
        924,z-720;903,z-759;865,z-773;833,z-785;854,z-834;
        859,z-864;822,z-858;807,z-845;785,z-794;767,z-793;726,z-809;
        687,z-821;660,z-812;670,z-795;696,z-782;745,z-763;775,z-748;
        777,z-740;772,z-730;752,z-731;711,z-740;664,z-747;664,z-722;
        690,z-708;779,z-694;777,z-676;760,z-671;721,z-671;707,z-659;
        727,z-644;734,z-640;768,z-634;819,z-631;836,z-632]);

    %右下角内圈
    [B_yxn_x,B_yxn_y]=myinv_fill_donghua2(a,sita,[832,z-718;863,z-703;875,z-703;879,z-713;875,z-727;850,z-737;
        834,z-739;830,z-722;832,z-718]);
    patch5=patch([B_yxw_x B_yxn_x],[B_yxw_y B_yxn_y],myblue);
    set(patch5,{'LineStyle'},{'none'}); %设置颜色和线

    %画圆
    ct=linspace(0,2*pi,1000);
    ox=0;
    oy=0;
    r0=375;
    %cx=657+366*cos(ct);
    %cy=z-582+366*sin(ct);
    cx1=ox+r0*cos(ct);
    cy1=oy+r0*sin(ct);
    %plot(cx1,cy1,'b-')
    cx2=ox+(r0+18)*cos(ct);
    cy2=oy+(r0+18)*sin(ct);
    %plot(cx2,cy2,'b-')

    cx3=ox+(r0+18+170)*cos(ct);
    cy3=oy+(r0+18+170)*sin(ct);
    %plot(cx3,cy3,'-','Color',myblue')

    cx4=ox+(r0+18+170+25)*cos(ct);
    cy4=oy+(r0+18+170+25)*sin(ct);
    %plot(cx4,cy4,'-','Color',myblue)


    patch1=patch(a*[cx1 cx2],a*[cy1 cy2],myblue);
    set(patch1,{'LineStyle'},{'none'}); %设置颜色和线

    patch2=patch(a*[cx3 cx4],a*[cy3 cy4],myblue);
    set(patch2,{'LineStyle'},{'none'}); %设置颜色和线
    axis(2.0*[-0.7*637 637 -0.7*590 590]);
    axis off;
    myframe=getframe;
    
    close all
end
