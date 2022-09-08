# from turtle import textinput
# from typing import Text

from googletrans import Translator

def translate(text,dest_language='zh-CN', src_language='id'):
    translator = Translator()# specify server url translator = Translator(service_urls=['translate.google.cn'])
    if isinstance(text,None):
        return None
    elif isinstance(text,str):
        trans_target=translator.translate(text,src=src_language,dest=dest_language)
        return trans_target.text
    elif isinstance(text,list):
        trans_target=[translator.translate(i,src=src_language,dest=dest_language).text for i in text]
        return trans_target
        

if __name__ == '__main__':
    text=["budayakan membaca sebelum melakukan pembeliancharger kamera fujifilm bc-45 charger fuji finepix cas instax mini 90 share sp 2 np-45a jv500 jz500produk ready ya gan / sisjadi langsung di order sajaakan segera kami proses dan kirimkwalitas barang okecharger untuk fujifilm :xp8020fd100fdz110dan masih banyak lagiuntuk seri yang tidak di sebutkan- fast respon jam kerja senin - jumat pukul 9.00 - 17.00 sabtu pukul 9.00 - 14.00 minggu atau hari libur off- mohon mencantumkan warna / gambar / varian pilihan dan cadangannya jika ada di kolom pemesanan yajika tidak mencantumkan pilihan maka akan kami kirimkan sesuai stok yang ready- pengiriman pada h + 1 malamnya- jika barang yang di pesan kosong dan tidak ada respon selama 1 x 24 jam maka akan di kirim secara random- periksa kembali nomor hp telp & alamat lengkapguna mempercepat dan memperlancar pengiriman- barang yang sudah di beli tidak bisa di kembalikan- gudang kita ada di dki jakarta jawa timur jawa tengah jawa barat dan lainnyajadi pengiriman tidak selalu sesuai alamat toko ini yaaamembeli = setuju.","produk terlaris - universal back corrector posture corrector protection back shoulder korektor postur tubuhpanduan membelilayak semua produk tersedia lanjutkan memesan jika anda menyukaipengiriman semua pesanan akan dikirim dalam waktu 24 jam dari bandung umumnya dibutuhkan 1 - 2 hari untuk tiba di bandung dan 2 - 7 hari untuk tiba di luar pulau jawaumpan balik mohon beri kami peringkat bintang 5 dengan komentar yang baik jika anda puas dengan produk dan layanan kamihadiah akan dikirim kepada anda secara gratis di pesanan anda berikutnya dengan menerima peringkat & komentar andafiturdirekomendasikan untuk anak-anak wanita dan pria kurusmembantu anda memperbaiki postur tubuh yang buruk mempertahankan bahu memperbaiki skoliosis dan kelengkungan tulang belakang yang salah bentuk lainnya1100 % baru2jadikan anda kecantikan yang elegan tidak peduli anda berjalan atau berdiri3sesuaikan bagian belakang dengan kekencangan yang paling cocok untuk diri anda4membantu memperbaiki postur buruk5membantu melatih tubuh anda untuk menjaga kembali bahu anda6membantu memperbaiki skoliosis dan kelengkungan tulang belakang malformasi lainnya7membantu meringankan postur terkait nyeri punggung nyeri bahu dan sakit kepalaspesifikasiwarna putihbahan : poliestercatatan : harap cuci tangan dan udara keringpaket termasuk1 pcs mendukung beltpemberitahuan1mohon diizinkan perbedaan 1 - 5 mm karena pengukuran manual2karena perbedaan antara monitor yang berbeda gambar mungkin tidak mencerminkan warna item yang sebenarnyasemoga anda menikmati waktu belanja anda ~terima kasih- catatan :fast respon jam kerja senin - jumat pukul 8.00 - 17.00 sabtu pukul 8.00 - 16.00 minggu offmohon mencantumkan warna / gambar / varian pilihan dan cadangannya jika ada di kolom pemesanan yaperiksa kembali nomor hp telp & alamat lengkapguna mempercepat dan memperlancar pengirimangudang kita ada di jakarta jawa dan lainnyajadi pengiriman tidak selalu sesuai alamat toko ini yaaaasemoga barang cocok# selamat berbelanja"]

    print(translate(text))






