import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def sendmailtolab(sub ,mytext):
    # ÉèÖÃsmtplibËùÐèµÄ²ÎÊý
    #ÏÂÃæµÄ·¢ŒþÈË£¬ÊÕŒþÈËÊÇÓÃÓÚÓÊŒþŽ«ÊäµÄ¡£
    smtpserver = 'smtp.163.com'
    username = 'zhangjunchenglab@163.com'
    password = 'DSHDVTAEIUWAWNVL'
    sender = 'zhangjunchenglab@163.com'
    # receiver='XXX@126.com'
    # ÊÕŒþÈËÎª¶àžöÊÕŒþÈË
    receiver = '736611728@qq.com'
    # receiver = 'liuyi19@mails.tsinghua.edu.cn'

    subject = sub
    # Íš¹ýHeader¶ÔÏó±àÂëµÄÎÄ±Ÿ£¬°üº¬utf-8±àÂëÐÅÏ¢ºÍBase64±àÂëÐÅÏ¢¡£ÒÔÏÂÖÐÎÄÃû²âÊÔok
    # subject = 'ÖÐÎÄ±êÌâ'
    # subject=Header(subject, 'utf-8').encode()

    # ¹¹ÔìÓÊŒþ¶ÔÏóMIMEMultipart¶ÔÏó
    # ÏÂÃæµÄÖ÷Ìâ£¬·¢ŒþÈË£¬ÊÕŒþÈË£¬ÈÕÆÚÊÇÏÔÊŸÔÚÓÊŒþÒ³ÃæÉÏµÄ¡£
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = username
    msg['To'] = receiver
    # ¹¹ÔìÎÄ×ÖÄÚÈÝ
    text = mytext
    text_plain = MIMEText(text, 'plain')
    msg.attach(text_plain)

    # ·¢ËÍÓÊŒþ
    smtp = smtplib.SMTP(smtpserver,25)
    smtp.connect('smtp.163.com')
    # ÎÒÃÇÓÃset_debuglevel(1)ŸÍ¿ÉÒÔŽòÓ¡³öºÍSMTP·þÎñÆ÷œ»»¥µÄËùÓÐÐÅÏ¢¡£
    #smtp.set_debuglevel(1)
    smtp.login(username, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()
