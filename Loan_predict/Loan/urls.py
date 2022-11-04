from django.urls import path,re_path
from Loan import views

urlpatterns =[
 path('', views.index, name='index'),
 re_path('predictLoan',views.predictLoan,name='pl'),

 #path('su/',views.predictLoan,name='index')
]