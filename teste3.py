import sys
from datetime import date
import numpy as np
import cv2

#CAPTURA DA DATA ATUAL
current_date = date.today()







### CONTADOR DE PRODUÇÃO ###

#Centro do contorno
def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy


#ATRIBUIÇÃO DO VÍDEO PARA UMA VARIÁVEL PARA ABRI-LO
cap = cv2.VideoCapture(0)

#CRIAÇÃO DE OBJETO PARA RETIRAR O FUNDO NA IMAGEM 
fgbg = cv2.createBackgroundSubtractorMOG2()

#FAZ A DETEÇÃO DO OBJETO NOS FRAMES FAZENDO UM CAMINHO POR ONDE ELE PASSOU
detects = []

#Posição da linha na vertical da esquerda para direita (está no meio do vídeo)
posL = 350
#quantidade pe pixels para começar a contar
offset = 100

#posição das linha x refere a posição na horizontal e y posição na vertical(está do tamanho da tela)
#LINHA CENTRAL
xy1 = (posL, 20)
xy2 = (posL, 470)
#LINHA ESQUERDA
xy3 = ((posL-offset), 20)
xy4 = ((posL-offset), 470)
#LINHA DIREITA
xy5 = ((posL+offset), 20)
xy6 = ((posL+offset), 470)


#LAÇO INFINITO
while 1:
    #1 - ATRIBUIÇÃO A UMA VARIÁVEL A LEITURA DO VIDEO
    ret, frame = cap.read()

    #2 - CONVERTER A IMAGEM PARA TONS DE CINZA
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("gray", gray)

    #3 - Retira a Mascara, aplica o método que identifica o que está sendo modificado do frame anterior
    fgmask = fgbg.apply(gray)
    #cv2.imshow("fgmask", fgmask)

    #4 - RETIRADA DOS NOISES DO FRAME - TIRA A SOMBRA QUE ESTÁ EM CINZA
        #200 REFERE-SE AO TOM DE CINZA QUE DEVE COMEÇAR A SER CONSIDERADO PARA FAZER A MUDANÇA PARA BRANCO QUE É O 255(SEGUNDO PARAMETRO SENDO PASSADO)
        #TRESH_BINARY É A FUNÇÃO QUE CONVERTE SOMENTE PARA PRETO OU BRACO 
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    #cv2.imshow("th", th)

    #5 - ELEMENTO ESTRUTURANTE - COMO O NUMPY SE UTILIZA DE FORMAS RETANGULAREM, FOI NECESSÁRIO A CRIAÇÃO DE ELEMNTOS ESTRUTURADOS PELA FUNÇÃO GETSTRUCTUTINGELEMENT DO OPENCV PARA SER UTILIZADO PELOS TRATAMENTOS MORFOLOGICOS A SEREM FEITOS. 
        #ESTRUTURA DE ELIPSE SENDO UTILIZADA
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #cv2.imshow("kernel", kernel)

    #6 - FUNÇÃO QUE ESTÁ TIRANDO OS RUÍDOS DA IMAGEM
        #INTERATION DETERMINA O TAMANHO MINIMO QUE O OBJETO EM MOVIMENTO PREVISA TER PARA NÃO SER CONSIDERADO NOISE
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations = 2)
    #cv2.imshow("opening", opening)

    #7 - FUNÇÃO DE DILATAÇÃO PARA QUE O OBJETO IDENTIFICADO TENHA SUAR MARGENS EXPANDIDAS
        #INTERATION = 20 REFERECE AO TAMANHO DA EXPANSÃO
    dilation = cv2.dilate(opening,kernel,iterations = 20)
    #cv2.imshow("dilation", dilation)

    #8 - FUNÇÃO QUE PREENCHE O OBJETO 
        #DEVIDO A DILATAÇÃO JÁ TER PRENCHIDO GRANDE PARTE DOS ESPAÇOS NO OBJETO, A QUANTIDADE DE INTERAÇÕES É MÍNIMA PARA NÃO INTERFERIR AINDA MAIS NO TAMANHO DOS OBJETOS
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 0)
    #cv2.imshow("closing", closing)
            

    #Apresenta as linhas na imagem
    #linha do centro
    cv2.line(frame,xy1,xy2,(255,0,0),3)
    #linha ESQUERDA
    cv2.line(frame,xy3,xy4,(255,255,0),2)
    #linha DIREITA
    cv2.line(frame,xy5,xy6,(255,255,0),2)


    #objetos que recebem atributos da função de contorno do OpenCv
    #CLOSING POIS É A IMAGEM QUE QUERO CONTORNAR
    #A FUNÇÃO RETR_TREE RECUPERA OS CONTORNOS E CRIA UMA HIERARQUIA FAMILIAR
    _, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #ID UTILIZADO PARA IDENTIFICAR CADA OBJETO NO FRAME CASO TENHA MAIS DE UM
    i = 0
    for cnt in contours:
        #determina o tamanho do retangulo envolta do objeto
        (x,y,w,h) = cv2.boundingRect(cnt)

        #calcula a área do objeto
        area = cv2.contourArea(cnt)
                
        #DETERMINA O TAMANHO MINIMO DA AREA DO OBJETO A SER DETECTADO
        if int(area) > 100 :
            #parametros para calcular o centro do retangulo
            centro = center(x, y, w, h)
            #Texto de numeração do retangulo do objeto
            cv2.putText(frame, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)

            #circulo no centro
            cv2.circle(frame, centro, 4, (0, 0,255), -1)
            #RETANDULO EM VOLTA DO OBJETO
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    
            #DETECTS É UMA VARIAVEL DE ARRAYS, PARA CADA ARRAY SERÃO ARMAZENADOS OS VALORES DO CENTRO DO OBJETO EM CADA FRAME
            if len(detects) <= i:
                #adiciona um array para o bjeto detectado para ser armazenado o valor dos seus centros por frame
                detects.append([])
                    
            #CENTRO[0] = X, POSL-OFFSET = LINHA DA ESQUERDA, POSL_OFFSET = LINHA DA DIREITA
            #SE SE POSIÇÃO DE X FOR MAIOR QUE A LINHA DA ESQUERDA E MENOS QUE A LINHA DA DIREITA O OBJETO ESTÁ DENTRO A ÁREA DE CONTAGEM A SER RASTREADO
            if centro[0]> posL-offset and centro[0] < posL+offset:
                detects[i].append(centro)
                #print(detects)
            else:
                detects[i].clear()
            i += 1

    #CASO NÃO TENHA NENHUM OBJETO NO FRAME
    if i == 0:
        detects.clear()

    i = 0
    #SE NÃO TIVER NENHUM CONTORNO, SEM OBJETOS PASSANDO, LIMPA A VARIAVEL DETECTS
    if len(contours) == 0:
        detects.clear()
    else:
        #ESTÁ PERCORRENDO CADA OBJETO DO ARRAY QUE OS DETECTOU 
        for detect in detects:
            #
            for (cod,posicao) in enumerate(detect):
                #SE PASSOU DA ESQUERDA PARA DIREIRA
                if detect[cod-1][0] < posL and posicao[0] > posL :
                    detect.clear()
                #     direita+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,0,255),5)
                    
                    
                     #print(total)
                    continue

    #textos que aparecem na tela 
    cv2.putText(frame, "TOTAL: "+str(total), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)

    #EXIBIÇÃO DO FRAME DO VÍDEO
    cv2.imshow("frame", frame)
    #CONDIÇÃO PARA PARAR O LAÇO INFINITO
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()